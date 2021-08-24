from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Customized import
from .model import NAOCell, RNNModel
from .utils import get_batch, repackage_hidden, embedded_dropout, mask2d, get_genotype, genotype_id_from_geno, PRIMITIVES
import gc

INITRANGE = 0.04

logger = logging.getLogger('nao_search.model')


class NAOCellSearch(NAOCell):

    def __init__(self, ninp, nhid, dropouth, dropoutx, drop_path=0.0, use_avg_leaf=False, num_intermediate_nodes=8,
                 handle_hidden_mode='None', emulate_ws_solutions=None):
        super(NAOCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, arch=None)

        self.emulate_ws_solutions = emulate_ws_solutions

        if self.emulate_ws_solutions is None:
            self._W0 = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))
            self._Ws = nn.ParameterList([
                nn.Parameter(torch.Tensor(4 * (i + 1) * nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for i in
                range(num_intermediate_nodes)
            ])

        else:
            self._W0s = nn.ParameterList([])
            self.w0s_dict = {}
            self._Ws_matrices = nn.ParameterList([])
            self.matrices_dict = {}

            for geno_id in range(0, self.emulate_ws_solutions):
                self.w0s_dict[geno_id] = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))
                self.matrices_dict[geno_id] = [
                    nn.Parameter(torch.Tensor(4 * (i + 1) * nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for i in
                    range(num_intermediate_nodes)
                ]

                self._W0s.append(self.w0s_dict[geno_id])
                self._Ws_matrices.extend(self.matrices_dict[geno_id])
                self.matrices_dict[geno_id] = nn.ParameterList(self.matrices_dict[geno_id])

            self._W0 = self.w0s_dict[0]
            self._Ws = self.matrices_dict[0]

        self.drop_path = drop_path
        self.use_avg_leaf = use_avg_leaf
        self.num_intermediate_nodes = num_intermediate_nodes
        self.handle_hidden_mode = handle_hidden_mode
        logger.info('nao cell search nodes: {}'.format(self.num_intermediate_nodes))

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def cell(self, x, h_prev, x_mask, h_mask, arch=None):
        assert arch is not None
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        # used for the leaf nodes logic
        concat = set([elem for elem in range(1, self.num_intermediate_nodes + 1)])

        states = [s0]
        for i in range(self.num_intermediate_nodes):
            pred, act = arch[2 * i], arch[2 * i + 1]
            s_prev = states[pred]
            w_start = (pred * 4 + act) * self.nhid
            w_end = w_start + self.nhid
            if self.training:
                self.apply_drop_path(self._Ws[i])
            w = self._Ws[i][w_start:w_end, :]
            if self.training:
                ch = (s_prev * h_mask).mm(w)
                self._Ws[i].requires_grad = True
            else:
                ch = s_prev.mm(w)
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(act)
            h = fn(h)
            s = s_prev + c * (h - s_prev)
            states.append(s)
            if pred in concat:
                concat.remove(pred)

        if self.use_avg_leaf:
            concat = sorted(list(concat))
        else:
            concat = range(self.num_intermediate_nodes + 1)[-self.num_intermediate_nodes:]

        # print('SEARCH OUTPUT NODES USED: {}'.format(concat))
        output = torch.mean(torch.stack([states[i] for i in concat], -1), dim=-1)

        if self.handle_hidden_mode == 'ACTIVATION':
            # logger.info('tanh used on output')
            output = F.tanh(output)
        return output

    def forward(self, inputs, hidden, arch=None):
        T, B = inputs.size(0), inputs.size(1)

        if self.training:
            x_mask = mask2d(B, inputs.size(2), keep_prob=1. - self.dropoutx)
            h_mask = mask2d(B, hidden.size(2), keep_prob=1. - self.dropouth)
        else:
            x_mask = h_mask = None

        hidden = hidden[0]
        hiddens = []

        if self.emulate_ws_solutions is not None:
            genotype = get_genotype(arch,
                                    self.num_intermediate_nodes,
                                    PRIMITIVES,
                                    use_avg_leaf=self.use_avg_leaf)
            genotype_id = genotype_id_from_geno(genotype)
            self._W0 = self.w0s_dict[genotype_id]
            self._Ws = self.matrices_dict[genotype_id]
            logger.info('weights matrices changed for the new geno_id: {}, genotype: {}'.format(genotype_id, genotype))

        for t in range(T):
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask, arch)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        return hiddens, hiddens[-1].unsqueeze(0)

    def apply_drop_path(self, w):
        if np.random.random() < self.drop_path:
            w.requires_grad = False


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=NAOCellSearch)
        self._args = args

    def clone(self):
        model_clone = RNNModelSearch(*self._args)
        for x, y in zip(model_clone.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_clone

    def _loss(self, hidden, input, target):
        log_prob, hidden_next = self(input, hidden, return_h=False)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
        return loss, hidden_next

    def arch(self):
        return self.last_arch

    def forward(self, input, hidden, arch=None, return_h=False):
        assert arch is not None
        self.last_arch = arch
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l], arch=arch)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden


def train(train_data, model, parallel_model, optimizer, params, epoch):
    assert params['batch_size'] % params['small_batch_size'] == 0, 'batch_size must be divisible by small_batch_size'
    '''
    if epoch == 1:
        logging.info('sampling before seed: {}'.format(np.random.random()))
        np.random.seed(1629)
        logging.info('sampling after seed: {}'.format(np.random.random()))
    '''
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    arch_pool = params['arch_pool']
    assert arch_pool is not None
    N = len(arch_pool)
    logger.info('N: {}'.format(N))

    hidden = [model.init_hidden(params['small_batch_size']) for _ in
              range(params['batch_size'] // params['small_batch_size'])]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:

        decision_seq = np.random.random()
        bptt = params['bptt'] if decision_seq < 0.95 else params['bptt'] / 2.
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / params['bptt']
        model.train()

        data, targets = get_batch(train_data, i, params['bptt'], seq_len=seq_len)
        sample_arch_index = np.random.randint(N)
        # logger.info('batch: {}, decision seq: {}, sampled arch {}'.format(batch, decision_seq, sample_arch_index))
        sample_arch = arch_pool[sample_arch_index]
        optimizer.zero_grad()

        start, end, s_id = 0, params['small_batch_size'], 0
        while start < params['batch_size']:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # assuming small_batch_size = batch_size so we don't accumulate gradients
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])

            # print('FORWARD TRAIN SEARCH!')
            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], sample_arch,
                                                                            return_h=True)
            # print('END FORWARD TRAIN SEARCH!')
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activation Regularization
            if params['alpha'] > 0:
                loss = loss + sum(
                    params['alpha'] * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(params['beta'] * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= params['small_batch_size'] / params['batch_size']
            total_loss += raw_loss.data * params['small_batch_size'] / params['batch_size']

            # logger.info('LOSS: {}'.format(loss.data[0]))
            # logger.info('RAW LOSS: {}'.format(raw_loss.data[0]))

            loss.backward()

            s_id += 1
            start = end
            end = start + params['small_batch_size']

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm(model.parameters(), params['clip'])
        optimizer.step()

        optimizer.param_groups[0]['lr'] = lr2
        if batch % params['log_interval'] == 0 and batch > 0:
            logger.info(parallel_model.arch())
            # logger.info('index of arch sampled: {}'.format(sample_arch_index))
            # logger.info('index of training: {}'.format(i))
            # logger.info('seq lenght of training: {}'.format(seq_len))
            logger.info('genotype: {}'.format(get_genotype(parallel_model.arch(),
                                                           params['num_intermediate_nodes'],
                                                           PRIMITIVES,
                                                           use_avg_leaf=params['use_avg_leaf'])))
            cur_loss = total_loss[0] / params['log_interval']
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                         'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // params['bptt'], optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / params['log_interval'], cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len


def evaluate(data_source, model, parallel_model, params, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    arch_pool = params['arch_pool']
    logger.info('Evaluating on {} archs'.format(len(arch_pool)))
    start_time = time.time()
    valid_score_list = []
    for arch in arch_pool:
        model.eval()
        hidden = model.init_hidden(batch_size)
        total_loss = 0
        for i in range(0, data_source.size(0) - 1, params['bptt']):
            # whether use random batch ?
            # data_source is in the format of [length, bs, ...]
            # for i in range(0, data_source.size(0) - 1, params['bptt']):
            # for i in range(1):
            # batch = np.random.randint(0, data_source.size(0) // params['bptt'])
            data, targets = get_batch(data_source, i, params['bptt'], evaluation=True)
            targets = targets.view(-1)
            log_prob, hidden = parallel_model(data, hidden, arch)
            loss = F.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data[0]

            total_loss += loss * len(data)
            hidden = repackage_hidden(hidden)

        valid_loss = total_loss / len(data_source)
        logger.info('valid ppl: {}, arch {}'.format(valid_loss, arch))
        valid_score_list.append(valid_loss)

    eval_time = time.time() - start_time
    mean_valid_score = np.mean(valid_score_list)
    logger.info(
        'Mean loss {:5.2f} | mean ppl {:8.2f} | time {:5.2f} secs'.format(mean_valid_score, np.exp(mean_valid_score),
                                                                          eval_time))
    return valid_score_list
