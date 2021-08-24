import time
import os, sys
import math
from collections import namedtuple

import numpy as np
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import gc

from nasws.dataset import dataloader
from nasws.rnn import model as model_module
from utils import batchify, get_batch, repackage_hidden, create_dir, save_checkpoint, create_viz_dir, to_item

# from visualize import plot

Rank = namedtuple('Rank', 'valid_ppl geno_id')


class EvaluationPhase:

    def __init__(self, args, genotype_id, genotype, test=False):
        self.args = args
        self.genotype_id = genotype_id
        self.genotype = genotype
        self.start_epoch = 1  # Keep track the starting epoch
        if test:
            self.initialize_test()
        else:
            self.initialize_run()

    def initialize_test(self):

        # Set the random seed manually for reproducibility.
        np.random.seed(self.args.evaluation_seed)
        torch.manual_seed(self.args.evaluation_seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.set_device(self.args.gpu)
                cudnn.benchmark = True
                cudnn.enabled = True
                torch.cuda.manual_seed_all(self.args.evaluation_seed)

        # preparing the test data
        self.test_batch_size = 1
        self.corpus = dataloader.Corpus(self.args.data)
        self.test_data = batchify(self.corpus.test, self.test_batch_size, self.args)

        # Load the best saved model.
        self.model = torch.load(os.path.join(self.args.test_dir, 'model.pt'))
        self.parallel_model = self.model.cuda()

    def initialize_run(self):
        """
        utility method for directories and plots
        :return:
        """
        if not self.args.continue_train:
            self.sub_directory_path = 'eval-{}_search_SEED_{}_eval_seed_{}_geno_id_{}'.format(self.args.save,
                                                                        self.args.seed,
                                                                        self.args.evaluation_seed,
                                                                        self.genotype_id,
                                                                        )
            self.exp_dir = os.path.join(self.args.main_path, self.sub_directory_path)
            create_dir(self.exp_dir)
        else: # continue to train
            assert self.args.resume_path is not None, FileNotFoundError("resume path cannot be empty.")
            self.sub_directory_path = self.args.resume_path
            self.exp_dir = os.path.join(self.args.main_path, self.sub_directory_path)
            if not os.path.exists(self.exp_dir):
                raise FileNotFoundError("Cannot found the train log under {}".format(self.exp_dir))


        if self.args.visualize:
            self.viz_dir_path = create_viz_dir(self.exp_dir)

        # preparation of the logger file
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.exp_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger("train_search")
        self.logger.addHandler(fh)

        # Set the random seed manually for reproducibility.
        self.logger.info('EVALUATION PHASE, USING SEED {}!'.format(self.args.evaluation_seed))
        np.random.seed(self.args.evaluation_seed)
        torch.manual_seed(self.args.evaluation_seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.set_device(self.args.gpu)
                cudnn.benchmark = True
                cudnn.enabled = True
                torch.cuda.manual_seed_all(self.args.evaluation_seed)

        self.lr = self.args.lr
        self.stored_loss = np.Inf
        self.best_val_loss = []

        # preparing the data
        if self.args.nhidlast < 0:
            self.args.nhidlast = self.args.emsize
        if self.args.small_batch_size < 0:
            self.args.small_batch_size = self.args.batch_size

        self.eval_batch_size = 10
        self.test_batch_size = 1
        self.corpus = dataloader.Corpus(self.args.data)

        self.train_data = batchify(self.corpus.train, self.args.batch_size, self.args)
        self.search_data = batchify(self.corpus.valid, self.args.batch_size, self.args)
        self.val_data = batchify(self.corpus.valid, self.eval_batch_size, self.args)
        self.test_data = batchify(self.corpus.test, self.test_batch_size, self.args)

        # Tensorboard writer for metrics
        tboard_dir = os.path.join(self.args.tboard_dir, self.sub_directory_path)
        self.writer = SummaryWriter(tboard_dir)

        # lists for storing metrics per epoch
        self.train_losses = []
        self.train_ppls = []
        self.valid_losses = []
        self.valid_ppls = []

        # initializing the model
        ntokens = len(self.corpus.dictionary)
        if self.args.use_pretrained:
            self.logger.info('PRETRAINED MODEL LOADED!')
            self.model = torch.load(os.path.join(self.args.pretrained_dir, 'model.pt'))
        else:
            self.logger.info('the search policy chose the genotype: {}, with genotype id: {}'.format(self.genotype,
                                                                                                       self.genotype_id))
            self.model = model_module.RNNModel(ntokens,
                                               self.args,
                                               self.genotype_id,
                                               genotype=self.genotype)

        # Continue train, rollback to previous best
        if self.args.continue_train:
            self.rollback_to_previous_best_model()
            self.start_epoch = self.epoch

        size = 0
        for p in self.model.parameters():
            self.logger.info('norm of model param: {}'.format(to_item(p.norm())))
            size += p.nelement()
        self.logger.info('param size: {}'.format(size))
        self.logger.info('initial genotype:')
        self.logger.info(self.model.genotype())

        # gpu handling
        if self.args.cuda:
            if self.args.single_gpu:
                self.logger.info('USING SINGLE GPU!')
                self.parallel_model = self.model.cuda()
            else:
                self.parallel_model = nn.DataParallel(self.model, dim=1).cuda()
        else:
            self.parallel_model = self.model

        total_params = sum(x.data.nelement() for x in self.model.parameters())
        self.logger.info('Args: {}'.format(self.args))
        self.logger.info('Model total parameters: {}'.format(total_params))

        # initializing the optimizer
        if self.args.continue_train:
            optimizer_state = torch.load(os.path.join(self.exp_dir, 'optimizer.pt'))
            if 't0' in optimizer_state['param_groups'][0]:
                self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.lr, t0=0, lambd=0.,
                                                  weight_decay=self.args.wdecay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.args.wdecay)
            self.optimizer.load_state_dict(optimizer_state)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.args.wdecay)

    def run(self):
        """
        main method for the train search handling all the epochs
        """
        try:

            # epochs logic starts here
            for epoch in range(self.start_epoch, self.args.epochs + 1):

                epoch_start_time = time.time()
                self.epoch = epoch
                self.logger.info('\n EPOCH {} STARTED.'.format(epoch))

                try:
                    # TRAINING LOGIC FOR ONE EPOCH
                    train_loss, train_ppl = self.train(epoch=epoch)
                    self.train_losses.append(train_loss)
                    self.train_ppls.append(train_ppl)

                except (KeyboardInterrupt, Exception) as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    self.logger.error(e)
                    self.rollback_to_previous_best_model()
                    continue

                # VALIDATION PART STARTS HERE
                if 't0' in self.optimizer.param_groups[0]:
                    self.tmp = {}
                    for prm in self.model.parameters():
                        if prm.grad is not None:
                            self.tmp[prm] = prm.data.clone()
                            prm.data = self.optimizer.state[prm]['ax'].clone()

                # validation at the end of the epoch
                val_loss, val_ppl = self.evaluate(self.val_data, batch_size=self.eval_batch_size)

                # tensorboard logs
                self.writer.add_scalar('validation_loss', val_loss, epoch)
                self.writer.add_scalar('validation_ppl', val_ppl, epoch)

                # saving val metrics
                self.valid_losses.append(val_loss)
                self.valid_ppls.append(val_ppl)

                # logs info
                self.logger.info('-' * 89)
                self.logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                                 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, val_ppl))
                self.logger.info('-' * 89)

                # call checkpoint if better model found
                if val_loss < self.stored_loss:
                    save_checkpoint(self.model, self.optimizer, epoch, self.exp_dir)
                    self.logger.info('Saving Normal!')
                    self.stored_loss = val_loss

                if 't0' in self.optimizer.param_groups[0]:
                    for prm in self.model.parameters():
                        if prm.grad is not None:
                            prm.data = self.tmp[prm].clone()

                # ASGD KICKS LOGIC
                if 't0' not in self.optimizer.param_groups[0] and (
                            len(self.best_val_loss) > self.args.nonmono and val_loss > min(self.best_val_loss[:-self.args.nonmono])):
                    self.logger.info('Switching!')
                    self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.lr, t0=0, lambd=0., weight_decay=self.args.wdecay)

                self.best_val_loss.append(val_loss)
                if len(self.best_val_loss) > self.args.nonmono:
                    self.logger.info('minimum fitness: {}'.format(min(self.best_val_loss[:-self.args.nonmono])))

            test_loss, test_ppl = self.test(model_path=self.exp_dir)

            return self.train_losses, self.train_ppls, self.valid_losses, self.valid_ppls, test_loss, test_ppl
        except (KeyboardInterrupt, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                logging.info('-' * 89)
                logging.info('Exiting from training early')
            test_loss, test_ppl = self.test(model_path=self.exp_dir)
            return self.train_losses, self.train_ppls, self.valid_losses, self.valid_ppls, test_loss, test_ppl

    def evaluate(self, data_source, batch_size=10):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, self.args.bptt):
            data, targets = get_batch(data_source, i, self.args, evaluation=True)
            targets = targets.view(-1)

            log_prob, hidden = self.parallel_model(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

            total_loss += loss * len(data)
            hidden = repackage_hidden(hidden)

        valid_loss = total_loss[0] / len(data_source)
        valid_ppl = math.exp(valid_loss)
        return valid_loss, valid_ppl

    def train(self, epoch):
        assert self.args.batch_size % self.args.small_batch_size == 0, 'bs must be divisible by small_batch_size'

        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0
        epoch_loss = 0
        start_time = time.time()
        ntokens = len(self.corpus.dictionary)

        # initializing first hidden state
        hidden = [self.model.init_hidden(self.args.small_batch_size) for _ in range(self.args.batch_size // self.args.small_batch_size)]

        batch, i, pop_index = 0, 0, 0
        while i < self.train_data.size(0) - 1 - 1:

            bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            seq_len = min(seq_len, self.args.bptt + self.args.max_seq_len_delta)

            lr2 = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt

            # preparing batch of data for training
            data, targets = get_batch(self.train_data, i, self.args, seq_len=seq_len)
            # IPython.embed()
            self.optimizer.zero_grad()

            start, end, s_id = 0, self.args.small_batch_size, 0
            while start < self.args.batch_size:
                cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden[s_id] = repackage_hidden(hidden[s_id])
                # hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

                # assuming small_batch_size = batch_size so we don't accumulate gradients
                self.optimizer.zero_grad()
                # hidden[s_id] = repackage_hidden(hidden[s_id])

                # forward pass
                log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = self.model(cur_data,
                                                                            hidden[s_id],
                                                                            return_h=True)

                # loss using negative-log-likelihood
                raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)
                loss = raw_loss

                # Activation Regularization
                if self.args.alpha > 0:
                    loss = loss + sum(
                        self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

                # if batch % self.args.log_interval == 0 and batch > 0:
                #     # for step in range(len(rnn_hs[0])):
                #     #    print("max hidden value of step " + str(step) + ': ' + str(rnn_hs[0][step].max()))
                #     self.logger.debug("max hidden value of all steps: " + str(to_item(rnn_hs[0].max())))

                # Temporal Activation Regularization (slowness)
                loss = loss + sum(self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                loss *= self.args.small_batch_size / self.args.batch_size

                total_loss += raw_loss.data * self.args.small_batch_size / self.args.batch_size
                epoch_loss += raw_loss.data * self.args.small_batch_size / self.args.batch_size

                loss.backward()

                s_id += 1
                start = end
                end = start + self.args.small_batch_size

                gc.collect()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)

            # applying the gradient updates
            self.optimizer.step()

            self.optimizer.param_groups[0]['lr'] = lr2

            if np.isnan(total_loss[0]):
                raise ValueError("Loss is nan")

            if batch % self.args.log_interval == 0 and batch > 0:

                cur_loss = total_loss[0] / self.args.log_interval

                elapsed = time.time() - start_time

                self.logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(self.train_data) // self.args.bptt, self.optimizer.param_groups[0]['lr'],
                                  elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss)))

                total_loss = 0
                start_time = time.time()

            batch += 1
            i += seq_len

        train_loss = epoch_loss[0] / batch
        train_ppl = math.exp(train_loss)

        # tensorboard train logs
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('train_ppl', train_ppl, epoch)

        return train_loss, train_ppl

    def test(self, model_path, use_logger=True):

        # Load the best saved model.
        self.model = torch.load(os.path.join(model_path, 'model.pt'))
        self.parallel_model = self.model.cuda()
        print('BEST MODEL LOADED, READY FOR TEST')

        # Run on test data.
        test_loss, test_ppl = self.evaluate(self.test_data, self.test_batch_size)

        print('=' * 89) if not use_logger else self.logger.info('=' * 89)
        if not use_logger:
            print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, test_ppl))
            print('=' * 89)
        else:
            self.logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                test_loss, test_ppl))
            self.logger.info('=' * 89)

        return test_loss, test_ppl

    def rollback_to_previous_best_model(self):
        self.model = torch.load(os.path.join(self.exp_dir, 'model.pt'))
        self.parallel_model = self.model.cuda()

        optimizer_state = torch.load(os.path.join(self.exp_dir, 'optimizer.pt'))
        if 't0' in optimizer_state['param_groups'][0]:
            self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.args.lr, t0=0, lambd=0.,
                                              weight_decay=self.args.wdecay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)
        self.optimizer.load_state_dict(optimizer_state)
        self.epoch = torch.load(os.path.join(self.exp_dir, 'misc.pt'))['epoch']
        self.logger.info('rolling back to the previous best model at epoch {}...'.format(self.epoch))
