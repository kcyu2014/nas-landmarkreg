"""
TO save the ranking change during the time.
compare with the ground-truth ranking.

"""
from copy import copy

import time
import math
from collections import namedtuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gc

import utils

from utils import get_batch, repackage_hidden
from .weight_sharing_ranking_random_on_batch import WeightSharingRandomRank

Rank = namedtuple('Rank', 'valid_ppl geno_id')


class FairSampleWeightSharingRandomRank(WeightSharingRandomRank):
    # number of sample.

    def sample_genotype(self):
        """
        FairNAS sample genotype.
        Does not change, because i need this sampled Architecture.
        :return:
        """
        # IPython.embed(header='sample genotype')
        genotype_id = np.random.randint(len(self.geno_ids))
        new_genotype = self.search_space.genotype_from_id(genotype_id=self.geno_ids[genotype_id])
        # selecting the current subDAG in our DAG to train
        self.model.change_genotype(genotype=new_genotype, genotype_id=self.geno_ids[genotype_id])
        return genotype_id, new_genotype

    def sample_ops(self):
        """
        Keep the topology not changing, only the sampled operations.
        :return:
        """
        # keep the genotype and only use the others
        choices = np.tile(np.arange(len(self.search_space.operations)), (self.search_space.intermediate_nodes, 1)).transpose()
        choices = np.apply_along_axis(np.random.permutation, 0, choices)
        # print('choices', choices)
        for i in range(len(self.search_space.operations)):
            _genotype = copy(self.model.genotype())
            for j in range(self.search_space.intermediate_nodes):
                _genotype.recurrent[j] = (self.search_space.operations[choices[i,j]], _genotype.recurrent[j][1])
            # print("new genotype during sample ops:", _genotype)
            _geno_id = self.search_space.genotype_id_from_geno(_genotype)
            yield self.model.change_genotype(genotype=_genotype, genotype_id=_geno_id)

    def train(self, epoch):
        assert self.args.batch_size % self.args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

        # Turn on training mode which enables dropout.
        total_loss = 0
        epoch_loss = 0
        tb_wpl_loss = 0
        start_time = time.time()

        # initializing first hidden state
        hidden = [self.model.init_hidden(self.args.small_batch_size) for _ in range(self.args.batch_size // self.args.small_batch_size)]

        batch, i, pop_index = 0, 0, 0

        # Looping the dataset.
        while i < self.train_data.size(0) - 1 - 1:

            # computing the genotype of the next particle
            # logging.debug('')
            genotype_id, genotype = self.sample_genotype()

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
            data, targets = get_batch(self.train_data, i, self.args, seq_len=seq_len)

            self.optimizer.zero_grad()

            start, end, s_id = 0, self.args.small_batch_size, 0
            num_architects = 0
            for _ in iter(self.sample_ops()):
                num_architects += 1
                # sample 4 different ops.
                while start < self.args.batch_size:
                    cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

                    # Starting each batch, we detach the hidden state from how it was previously produced.
                    # If we didn't, the model would try backpropagating all the way to start of the dataset.
                    hidden[s_id] = repackage_hidden(hidden[s_id])
                    # hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

                    # assuming small_batch_size = batch_size so we don't accumulate gradients
                    # self.optimizer.zero_grad()
                    # hidden[s_id] = repackage_hidden(hidden[s_id])
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

                    loss *= self.args.small_batch_size / self.args.batch_size / 4
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

                total_loss = 0
                start_time = time.time()

            batch += 1
            i += seq_len

        self.writer.add_scalar('train_loss', utils.to_item(epoch_loss) / batch, epoch)
        self.writer.add_scalar('train_ppl', math.exp(utils.to_item(epoch_loss) / batch), epoch)
