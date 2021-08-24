import os, sys, glob
import time
import math
import numpy as np
import torch
# after import torch, add this
import utils

torch.backends.cudnn.benchmark=True
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import gc

from nasws.dataset import dataloader
from .architect import Architect
from . import darts_model_search as mod_search
from .utils import batchify, get_batch, create_exp_dir, save_checkpoint
from utils import repackage_hidden
from nasws.rnn.general_search_policies import RNNSearchPolicy


class DartsSearchPolicy(RNNSearchPolicy):

    def __init__(self, args):
        super(DartsSearchPolicy, self).__init__(args=args)

        self.initialize_search()

    def initialize_search(self):
        if self.args.nhidlast < 0:
            self.args.nhidlast = self.args.emsize
        if self.args.small_batch_size < 0:
            self.args.small_batch_size = self.args.batch_size

        self.search_dir = 'search_{}_second_order_{}_SEED_{}_{}'.format(self.args.save,
                                                                        self.args.unrolled,
                                                                        self.args.seed,
                                                                        time.strftime("%Y%m%d-%H%M%S")
                                                                        )

        self.search_dir = os.path.join(self.args.main_path, self.search_dir)

        create_exp_dir(self.search_dir, scripts_to_save=glob.glob('*.py'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.search_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger("darts_search")
        self.logger.addHandler(fh)

        # Set the random seed manually for reproducibility.
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.set_device(self.args.gpu)
                cudnn.benchmark = True
                cudnn.enabled = True
                torch.cuda.manual_seed_all(self.args.seed)

        self.corpus = dataloader.Corpus(self.args.data)

        self.eval_batch_size = 10
        self.test_batch_size = 1

        self.train_data = batchify(self.corpus.train, self.args.batch_size, self.args)
        self.search_data = batchify(self.corpus.valid, self.args.batch_size, self.args)
        self.val_data = batchify(self.corpus.valid, self.eval_batch_size, self.args)
        self.test_data = batchify(self.corpus.test, self.test_batch_size, self.args)

        # Tensorboard writer
        self.writer = SummaryWriter('runs/tensor_log_' + self.search_dir)

        self.ntokens = len(self.corpus.dictionary)

        # initializing the model

        self.model = mod_search.RNNModelSearch(self.ntokens, self.args.emsize, self.args.nhid, self.args.nhidlast,
                                               self.args.dropout, self.args.dropouth, self.args.dropoutx,
                                               self.args.dropouti, self.args.dropoute, self.args.num_intermediate_nodes,
                                               self.args.handle_hidden_mode)

        size = 0
        for p in self.model.parameters():
            size += p.nelement()
        self.logger.info('param size: {}'.format(size))
        self.logger.info('initial genotype:')
        self.logger.info(self.model.genotype())

        if self.args.cuda:
            if self.args.single_gpu:
                self.parallel_model = self.model.cuda()
            else:
                self.parallel_model = nn.DataParallel(self.model, dim=1).cuda()
        else:
            self.parallel_model = self.model

        # initializing the architecture search optimizer
        self.architect = Architect(self.parallel_model, self.args)

        boo_val = self.args.unrolled == True
        print("BOOLEAN UNROLLED", boo_val)

        total_params = sum(x.data.nelement() for x in self.model.parameters())
        self.logger.info('Args: {}'.format(self.args))
        self.logger.info('Model total parameters: {}'.format(total_params))

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
        return utils.to_item(total_loss) / len(data_source)

    def train(self):
        assert self.args.batch_size % self.args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

        # Turn on training mode which enables dropout.
        self.total_loss = 0
        self.epoch_loss = 0
        self.start_time = time.time()
        self.ntokens = len(self.corpus.dictionary)

        # initializing first hidden state
        hidden = [self.model.init_hidden(self.args.small_batch_size) for _ in
                  range(self.args.batch_size // self.args.small_batch_size)]
        hidden_valid = [self.model.init_hidden(self.args.small_batch_size) for _ in
                        range(self.args.batch_size // self.args.small_batch_size)]

        batch, i = 0, 0

        while i < self.train_data.size(0) - 1 - 1:

            bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            # seq_len = max(5, int(np.random.normal(bptt, 5)))
            # # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
            seq_len = int(bptt)

            lr2 = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt

            # training mode activated
            self.model.train()

            # preparing batch of data from validation for the architecture optimizer step
            data_valid, targets_valid = get_batch(self.search_data, i % (self.search_data.size(0) - 1), self.args)

            # preparing batch of data for training
            data, targets = get_batch(self.train_data, i, self.args, seq_len=seq_len)

            self.optimizer.zero_grad()

            start, end, s_id = 0, self.args.small_batch_size, 0
            while start < self.args.batch_size:
                cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
                cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:,
                                                                               start: end].contiguous().view(-1)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden[s_id] = repackage_hidden(hidden[s_id])
                hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

                # architecture optimizer step, updating the alphas weights
                hidden_valid[s_id], grad_norm = self.architect.step(hidden[s_id],
                                                                    cur_data,
                                                                    cur_targets,
                                                                    hidden_valid[s_id],
                                                                    cur_data_valid,
                                                                    cur_targets_valid,
                                                                    self.optimizer,
                                                                    self.args.unrolled)

                # assuming small_batch_size = batch_size so we don't accumulate gradients
                self.optimizer.zero_grad()
                hidden[s_id] = repackage_hidden(hidden[s_id])

                # forward pass
                log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = self.parallel_model(cur_data, hidden[s_id],
                                                                                     return_h=True)

                # loss using negative-log-likelihood
                raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

                loss = raw_loss
                # Activiation Regularization
                if self.args.alpha > 0:
                    loss = loss + sum(
                        self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

                # Temporal Activation Regularization (slowness)
                loss = loss + sum(self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                loss *= self.args.small_batch_size / self.args.batch_size
                self.total_loss += raw_loss.data * self.args.small_batch_size / self.args.batch_size

                # accumulate the loss of the epoch for tensorboard log
                self.epoch_loss += raw_loss.data * self.args.small_batch_size / self.args.batch_size
                loss.backward()

                s_id += 1
                start = end
                end = start + self.args.small_batch_size

                gc.collect()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip)
            self.optimizer.step()

            # total_loss += raw_loss.data
            self.optimizer.param_groups[0]['lr'] = lr2
            if batch % self.args.log_interval == 0 and batch > 0:
                self.logger.info('current GENOTYPE: {}'.format(self.parallel_model.genotype()))
                cur_loss = utils.to_item(self.total_loss) / self.args.log_interval
                elapsed = time.time() - self.start_time
                self.logger.info('|SEARCH PHASE RUNNING| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                             'loss {:5.2f} | ppl {:8.2f}'.format(
                    self.epoch, batch, len(self.train_data) // self.args.bptt, self.optimizer.param_groups[0]['lr'],
                                  elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss)))
                self.total_loss = 0
                self.start_time = time.time()
            batch += 1
            i += seq_len
        self.writer.add_scalar('train_loss', cur_loss, self.epoch)
        self.writer.add_scalar('train_ppl', math.exp(cur_loss), self.epoch)

    def run(self):

        # Loop over epochs.
        lr = self.args.lr
        best_val_loss = []
        stored_loss = 100000000
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)

        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            self.epoch = epoch
            self.logger.info('EPOCH {} STARTED'.format(self.epoch))

            # training pipeline for one epoch
            self.train()

            # validation at the end of the epoch
            val_loss = self.evaluate(self.val_data, self.eval_batch_size)
            self.writer.add_scalar('validation_loss', val_loss, epoch)
            self.writer.add_scalar('validation_ppl', math.exp(val_loss), epoch)

            # logs info
            self.logger.info('-' * 89)
            self.logger.info('|SEARCH PHASE RUNNING| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                    val_loss, math.exp(val_loss)))
            self.logger.info('-' * 89)

            # call checkpoint if better model found
            if val_loss < stored_loss:
                save_checkpoint(self.model, self.optimizer, epoch, self.search_dir)
                self.logger.info('Saving Normal!')
                stored_loss = val_loss

            best_val_loss.append(val_loss)

        genotype = self.model.genotype()
        genotype_id = self.search_space.genotype_id_from_geno(genotype=genotype)

        self.logger.info("best genotype end search: {}, with id: {}".format(genotype, genotype_id))
        return genotype_id, genotype
