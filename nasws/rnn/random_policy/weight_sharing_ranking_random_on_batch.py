"""
TO save the ranking change during the time.
compare with the ground-truth ranking.

"""
import time
import os
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import operator

import gc

from nasws.dataset import dataloader
from nasws.rnn import model as model_module
import utils

from utils import batchify, get_batch, repackage_hidden, save_checkpoint
from nasws import SearchPolicy

Rank = namedtuple('Rank', 'valid_ppl geno_id')


class WeightSharingRandomRank(SearchPolicy):

    def __init__(self, args):
        super(WeightSharingRandomRank, self).__init__(args=args)
        self.model_fn = model_module.RNNModel
        self.initialize_run()

    def sample_genotype(self):
        """
        Sample and apply the new genotype.
        :return:
        """
        genotype_id = np.random.randint(len(self.geno_ids))
        new_genotype = self.search_space.genotype_from_id(genotype_id=self.geno_ids[genotype_id])
        # selecting the current subDAG in our DAG to train
        self.model.change_genotype(genotype=new_genotype, genotype_id=self.geno_ids[genotype_id])
        return genotype_id, new_genotype

    def evaluate(self, epoch, data_source, fitnesses_dict, batch_size=10):
        """
        Evaluates all the current architectures in the population
        """

        particles_indices = [i for i in range(len(self.geno_ids))]
        np.random.shuffle(particles_indices)

        avg_particle_loss = 0
        avg_particle_ppl = 0

        genotypes_fit_dict = {}

        # rank dict for the possible solutions
        genotypes_rank = {}

        for particle_id in particles_indices:

            geno_id = self.geno_ids[particle_id]

            # computing the genotype of the next particle
            new_genotype = self.search_space.genotype_from_id(genotype_id=geno_id)

            # selecting the current subDAG in our DAG to train
            self.model.change_genotype(genotype=new_genotype, genotype_id=geno_id)
            self.logger.debug('evaluate the geno id: {}'.format(geno_id))
            avg_valid_loss, avg_valid_ppl = self.validate_model(current_model=self.model,
                                                                data_source=data_source,
                                                                current_geno_id=geno_id,
                                                                current_genotype=self.model.genotype(),
                                                                batch_size=batch_size)

            avg_particle_loss += avg_valid_loss
            avg_particle_ppl += avg_valid_ppl

            # saving the particle fit in our dictionaries
            fitnesses_dict[particle_id] = avg_valid_ppl
            genotypes_fit_dict[geno_id] = avg_valid_ppl
            gen_key = str(self.model.genotype().recurrent)
            genotypes_rank[gen_key] = Rank(avg_valid_ppl, geno_id)

        # save the ranking, according to their GENOTYPE but not particle id
        rank_gens = sorted(genotypes_rank.items(), key=operator.itemgetter(1))
        self.ranking_per_epoch[epoch] = rank_gens

        self.logger.info('VALIDATION RANKING OF PARTICLES')
        for pos, elem in enumerate(rank_gens):
            temp_gen = elem[0]
            temp_fit = elem[1].valid_ppl
            temp_id = elem[1].geno_id

            self.logger.info('particle gen id: {}, ppl: {}, gen: {}'.format(
                temp_id,
                temp_fit,
                temp_gen)
            )

        return fitnesses_dict, avg_particle_loss / len(self.cluster_dict), avg_particle_ppl / len(self.cluster_dict)

    def train(self, epoch):
        # TODO detach this to share across other models!
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
            genotype_id, genotype = self.sample_genotype()
            # logging.debug('')

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

                total_loss = 0
                start_time = time.time()

            batch += 1
            i += seq_len

        self.writer.add_scalar('train_loss', utils.to_item(epoch_loss) / batch, epoch)
        self.writer.add_scalar('train_ppl', math.exp(utils.to_item(epoch_loss) / batch), epoch)

    def initialize_model(self, ntokens, genotype_id, genotype):
        # initializing the model
        if self.args.use_pretrained:
            self.logger.info('PRETRAINED MODEL LOADED!')
            self.model = torch.load(os.path.join(self.args.pretrained_dir, 'model.pt'))

        else:
            self.model = self.model_fn(ntokens,
                                       self.args,
                                       genotype_id,
                                       genotype=genotype)

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

        total_params = sum(x.data.nelement() for x in self.model.parameters())
        self.logger.info('Args: {}'.format(self.args))
        self.logger.info('Model total parameters: {}'.format(total_params))

    def run(self):
        """
        main method for the train search handling all the epochs
        """
        if self.args.nhidlast < 0:
            self.args.nhidlast = self.args.emsize
        if self.args.small_batch_size < 0:
            self.args.small_batch_size = self.args.batch_size

        # same eval batch size of NAO
        self.eval_batch_size = 256
        self.test_batch_size = 1
        self.corpus = dataloader.Corpus(self.args.data)

        self.train_data = batchify(self.corpus.train, self.args.batch_size, self.args)
        self.search_data = batchify(self.corpus.valid, self.args.batch_size, self.args)
        self.val_data = batchify(self.corpus.valid, self.eval_batch_size, self.args)
        self.test_data = batchify(self.corpus.test, self.test_batch_size, self.args)

        self.logger.info('train len: {}'.format(self.train_data.size()))
        self.logger.info('val len: {}'.format(self.val_data.size()))

        # Tensorboard writer
        tboard_dir = os.path.join(self.args.tboard_dir,self.sub_directory_path)
        self.writer = SummaryWriter(tboard_dir)

        ntokens = len(self.corpus.dictionary)

        # initialize the swarm
        if self.args.genotype_start is not None:
            self.geno_ids = [id for id in range(self.args.genotype_start, self.args.genotype_end)]
        else:
            self.geno_ids = [id for id in range(self.search_space.num_solutions)]

        # initial genotype
        genotype_id = 0
        genotype = self.search_space.genotype_from_id(genotype_id)

        self.particles_fitnesses = {}

        self.initialize_model(ntokens, genotype_id, genotype)

        # Loop over epochs.
        self.lr = self.args.lr
        self.best_val_loss = []
        stored_loss = 100000000

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.args.wdecay)

        # epochs logic starts here
        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            self.epoch = epoch
            self.logger.info('\n EPOCH {} STARTED.'.format(epoch))

            # update the clusters dictionary of unique solutions
            self.cluster_dict = set(self.geno_ids)

            # TRAINING LOGIC FOR ONE EPOCH
            self.train(epoch=epoch)

            # VALIDATION PART STARTS HERE
            if epoch % self.args.save_every_epoch == 0:
                if 't0' in self.optimizer.param_groups[0]:
                    self.tmp = {}
                    for prm in self.model.parameters():
                        if prm.grad is not None:
                            self.tmp[prm] = prm.data.clone()
                            prm.data = self.optimizer.state[prm]['ax'].clone()
                    self.logger.info('CLONING Weights SUCCESSFULL')

                # validation at the end of the epoch
                self.particles_fitnesses, val_loss, val_ppl = self.evaluate(epoch,
                                                                            self.val_data,
                                                                            fitnesses_dict=self.particles_fitnesses,
                                                                            batch_size=self.eval_batch_size)
                self.save_results(epoch, rank_details=True)

                # logs info
                self.logger.info('-' * 89)
                self.logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                       math.log(val_ppl), val_ppl))
                self.logger.info('-' * 89)
                if self.writer is not None:
                    self.writer.add_scalar('valid_ppl', val_ppl, epoch)
                    self.writer.add_scalar('valid_loss', val_loss, epoch)

                # call checkpoint if better model found
                if val_loss < stored_loss:
                    save_checkpoint(self.model, self.optimizer, epoch, self.exp_dir)
                    self.logger.info('Saving Normal!')
                    stored_loss = val_loss

                if 't0' in self.optimizer.param_groups[0]:
                    for prm in self.model.parameters():
                        if prm.grad is not None:
                            prm.data = self.tmp[prm].clone()

                # ASGD KICKS LOGIC
                if 't0' not in self.optimizer.param_groups[0] and (
                            len(self.best_val_loss) > self.args.nonmono and val_ppl > min(self.best_val_loss[:-self.args.nonmono])):
                    self.logger.info('Switching!')
                    self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.lr, t0=0, lambd=0., weight_decay=self.args.wdecay)

                self.best_val_loss.append(val_ppl)
                if len(self.best_val_loss) > self.args.nonmono:
                    self.logger.info('minimum fitness: {}'.format(min(self.best_val_loss[:-self.args.nonmono])))

        # always return the first architecture! should not be used as evaluation.
        return 0, self.search_space.genotype_from_id(0)

