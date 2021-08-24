import logging

import time
import math
import os
import gc
import torch
import torch.nn as nn
import numpy as np

import utils
from nasws.rnn.random_policy.weight_sharing_ranking_random_on_batch import WeightSharingRandomRank
from .soft_weight_sharing_ranking_random_on_batch import SoftWeightSharingRandomRank
from .darts_architect import SoftWSArchitect
# import nasws.rnn.softws.soft_weight_sharing_model as model_module
from .soft_weight_sharing_model_search import RNNModelSoftWSSearch


class SoftWeightSharingRandomRankWithSearchTracking(SoftWeightSharingRandomRank):
    """
    Potentially, add also the WPL into the Architect.
    """
    genotype_id_sequence = []

    def sample_genotype(self):
        genotype, genotype_id = super(SoftWeightSharingRandomRankWithSearchTracking, self).sample_genotype()
        logging.info('sample a new genotype ', genotype_id)
        self.genotype_id_sequence.append(genotype_id)

    def save_results(self, epoch, rank_details=False):
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
            'train_genoid_sequence': self.genotype_id_sequence
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details)


class SoftWeightSharingRandomRankWithSearch(SoftWeightSharingRandomRank):

    # This is a class that I visualize all the possible results.
    
    def __init__(self, args):
        super(WeightSharingRandomRank, self).__init__(args=args)
        print(">>> Soft-weight sharing enabled! <<<")
        self.model_fn = RNNModelSoftWSSearch
        self.initialize_run()

        # model is not yet created, use the initialize model to create
        # shall be created later
        self.architect = None
        self.model = None

    def initialize_model(self, ntokens, genotype_id, genotype):
        # add search space into args.
        self.args.search_space = self.search_space
        # initializing the model
        if self.args.use_pretrained:
            self.logger.info('PRETRAINED MODEL LOADED!')
            self.model = torch.load(os.path.join(self.args.pretrained_dir, 'model.pt'))

        else:
            self.model = self.model_fn(ntokens,
                                       self.args,
                                       genotype_id,
                                       genotype=genotype)
            # should pass these into


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

        # Add the Architect.
        if self.args.softws_policy == 'darts':
            logging.info("DARTS policy for soft-weight-sharing.")
            self.architect = SoftWSArchitect(self.model, self.args)
        elif self.args.softws_policy == 'nao':
            logging.info('NAO policy for soft-weight-sharing.')
            self.architect = None
        else:
            raise NotImplementedError("{} policy not supported! ".format(self.args.softws_policy))

        total_params = sum(x.data.nelement() for x in self.model.parameters())
        self.logger.info('Args: {}'.format(self.args))
        self.logger.info('Model total parameters: {}'.format(total_params))

    def train(self, epoch):
        assert self.args.batch_size % self.args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

        # Turn on training mode which enables dropout.
        total_loss = 0
        epoch_loss = 0
        tb_wpl_loss = 0
        start_time = time.time()

        # initializing first hidden state
        hidden = [self.model.init_hidden(self.args.small_batch_size) for _ in
                  range(self.args.batch_size // self.args.small_batch_size)]
        hidden_valid = [self.model.init_hidden(self.args.small_batch_size) for _ in
                        range(self.args.batch_size // self.args.small_batch_size)]

        batch, i, pop_index = 0, 0, 0

        # Looping the dataset.
        while i < self.train_data.size(0) - 1 - 1:

            # computing the genotype of the next particle
            # genotype_id = np.random.randint(len(self.geno_ids))
            # new_genotype = self.search_space.genotype_from_id(genotype_id=self.geno_ids[genotype_id])
            #
            # selecting the current subDAG in our DAG to train
            # self.model.change_genotype(genotype=new_genotype, genotype_id=self.geno_ids[genotype_id])
            self.sample_genotype()

            bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            # seq_len = max(5, int(np.random.normal(bptt, 5)))
            # # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, self.args.bptt + self.args.max_seq_len_delta)
            seq_len = int(bptt)

            lr2 = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt

            # training mode activated
            self.model.train()

            # preparing batch of data for training
            data, targets = utils.get_batch(self.train_data, i, self.args, seq_len=seq_len)

            # preparing batch of validation for updating weights.
            data_valid, targets_valid = utils.get_batch(self.search_data, i % (self.search_data.size(0) - 1), self.args)

            self.optimizer.zero_grad()

            start, end, s_id = 0, self.args.small_batch_size, 0
            while start < self.args.batch_size:
                cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
                cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:,
                                                                               start: end].contiguous().view(-1)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden[s_id] = utils.repackage_hidden(hidden[s_id])
                hidden_valid[s_id] = utils.repackage_hidden(hidden_valid[s_id])

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
                hidden[s_id] = utils.repackage_hidden(hidden[s_id])

                # forward pass
                log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = self.model(cur_data,
                                                                            hidden[s_id],
                                                                            return_h=True)

                # loss using negative-log-likelihood
                raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)
                loss = raw_loss

                try:
                    # Activation Regularization
                    if self.args.alpha > 0:
                        loss = loss + sum(
                            self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

                    if batch % self.args.log_interval == 0 and batch > 0:
                        # for step in range(len(rnn_hs[0])):
                        #    print("max hidden value of step " + str(step) + ': ' + str(rnn_hs[0][step].max()))
                        self.logger.info("max hidden value of all steps: " + str(utils.to_item(rnn_hs[0].max())))

                    # Temporal Activation Regularization (slowness)
                    loss = loss + sum(self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

                except:
                    self.logger.info("max hidden value of all steps: " + str(utils.to_item(rnn_hs[0].max())))
                    print('RNN_HS: {}'.format(rnn_hs))
                    self.logger.info("genotype who caused the error:  ")
                    self.logger.info(self.model.genotype())
                    for name_param, param in self.model.rnns[0].named_parameters():
                        self.logger.info("param name: " + str(name_param))
                        self.logger.info("max value in the param matrix: " + str(param.max()))
                    raise

                loss *= self.args.small_batch_size / self.args.batch_size
                total_loss += utils.to_item(raw_loss) * self.args.small_batch_size / self.args.batch_size
                epoch_loss += utils.to_item(raw_loss) * self.args.small_batch_size / self.args.batch_size
                loss.backward()

                s_id += 1
                start = end
                end = start + self.args.small_batch_size

                gc.collect()

            # applying the gradient updates
            utils.clip_grad_norm(self.model.parameters(), self.args.clip)
            self.optimizer.step()

            self.optimizer.param_groups[0]['lr'] = lr2

            if batch % self.args.log_interval == 0 and batch > 0:
                self.logger.info(self.model.genotype())
                # print(F.softmax(parallel_model.weights, dim=-1))
                cur_loss = total_loss / self.args.log_interval
                elapsed = time.time() - start_time
                self.logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                 'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(self.train_data) // self.args.bptt, self.optimizer.param_groups[0]['lr'],
                                  elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss)))
                # self.logger.info('Current storing dict for Soft-WS: ')
                # utils.dumpclean(self.model.softws_parameters(), self.logger.info)
                total_loss = 0
                start_time = time.time()

            batch += 1
            i += seq_len

        # add train loss into this.
        self.writer.add_scalar('train_loss', utils.to_item(epoch_loss) / batch, epoch)
        self.writer.add_scalar('train_ppl', math.exp(utils.to_item(epoch_loss) / batch), epoch)
