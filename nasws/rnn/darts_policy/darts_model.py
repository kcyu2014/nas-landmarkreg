import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mask2d
from utils import LockedDropout
from utils import embedded_dropout
from torch.autograd import Variable
import numpy as np
import logging

from utils import sigmoid

INITRANGE = 0.04


class DARTSCell(nn.Module):
    """
    torch module class representing the DARTS cell
    """

    def __init__(self, ninp, nhid, dropouth, dropoutx, genotype, num_intermediate_nodes, handle_hidden_mode=None):
        """
        :param ninp: input size (word embedding size)
        :param nhid: hidden state size (number of hidden units of the FC layer in a rnn cell)
        :param dropouth: dropout rate for hidden
        :param dropoutx: dropout rate for input
        :param genotype: string representation of the cell (description of the edges involved)
        """
        super(DARTSCell, self).__init__()
        self.nhid = nhid
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.genotype = genotype
        self.num_intermediate_nodes = num_intermediate_nodes
        self.handle_hidden_mode = handle_hidden_mode

        # IMPORTANT: genotype is None when doing arch search
        # the steps are equal to the number of intermediate nodes
        # in the cell (default 4 nodes for cnns, 8 nodes for rnn)
        steps = len(self.genotype.recurrent) if self.genotype is not None else self.num_intermediate_nodes

        # initializing the first weight matrix between input x and the hidden layer
        self._W0 = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))

        # initializing the weight matrices used towards the intermediate nodes (range of steps)
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.Tensor(nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for _ in range(steps)
                                    ])

    def forward(self, inputs, hidden):
        T, B = inputs.size(0), inputs.size(1)

        if self.training:
            x_mask = mask2d(B, inputs.size(2), keep_prob=1. - self.dropoutx)
            h_mask = mask2d(B, hidden.size(2), keep_prob=1. - self.dropouth)
        else:
            x_mask = h_mask = None

        hidden = hidden[0]
        hiddens = []
        clipped_num = 0
        max_clipped_norm = 0

        # forward pass through time in the cell, T is the sequence length
        for t in range(T):
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask)

            if self.handle_hidden_mode == 'NORM':

                hidden_norms = hidden.norm(dim=-1)
                max_norm = 30.0
                if hidden_norms.data.max() > max_norm:
                    # Just directly use the torch slice operations
                    # in PyTorch v0.4.
                    #
                    # This workaround for PyTorch v0.3.1 does everything in numpy,
                    # because the PyTorch slicing and slice assignment is too
                    # flaky.
                    print('HIGH NORM: {}'.format(hidden_norms.data.max()))
                    hidden_norms = hidden_norms.data.cpu().numpy()

                    clipped_num += 1
                    if hidden_norms.max() > max_clipped_norm:
                        max_clipped_norm = hidden_norms.max()

                    clip_select = hidden_norms > max_norm
                    clip_norms = hidden_norms[clip_select]

                    mask = np.ones(hidden.size())
                    normalizer = max_norm / clip_norms
                    normalizer = normalizer[:, np.newaxis]

                    mask[clip_select] = normalizer
                    hidden *= torch.autograd.Variable(
                        torch.FloatTensor(mask).cuda(), requires_grad=False)
                    print('NEW HIDDEN NORM: {}'.format(hidden.norm(dim=-1).data.max()))

            if self.handle_hidden_mode == 'ACTIVATION_ON_STEP':
                # avoid the explosion of the hidden state by forcing the value in [-1, 1]
                hidden = F.tanh(hidden)

            # saving the hidden output for each step
            hiddens.append(hidden)

        if clipped_num > 0:
            print('clipped {} hidden states in one forward '.format(clipped_num))
            print('max clipped hidden state norm: {}'.format(max_clipped_norm))

        # creating a tensor from the list of hiddens in order to have the elements stacked
        hiddens = torch.stack(hiddens)

        # return the stack of hidden outputs and the hidden output of the last step
        return hiddens, hiddens[-1].unsqueeze(0)

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

    def _get_activation(self, name):
        if name == 'tanh':
            f = F.tanh
        elif name == 'relu':
            f = F.relu
        elif name == 'sigmoid':
            f = sigmoid
        elif name == 'identity':
            f = lambda x: x
        else:
            raise NotImplementedError
        return f

    def cell(self, x, h_prev, x_mask, h_mask):
        """
        forwards inside the cell of our model
        :param x: input
        :param h_prev: hidden of previous step
        :param x_mask: mask for input dropout
        :param h_mask: mask for hidden dropout
        :return: the hidden output of the current step
        """

        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        # states contains the nodes computed as described in the paper,
        # that means, the sum of output of the operations of the incoming
        # edges. If genotype defined, there is only one incoming edge
        # as a constraint described in the paper.
        states = [s0]

        # IMPORTANT: genotype is None when doing arch search
        # "i" is the index of the next intermediate node,
        # "name" is the label of the activation function,
        # "pred" is the index of the previous node, so the edge will be pred -->name--->i
        for i, (name, pred) in enumerate(self.genotype.recurrent):

            # taking the previous state using its index
            s_prev = states[pred]

            # applying dropout masks if training.
            # computing the matrix mul between the previous output
            # and the weights of the current node "i" (FC layer)
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])
            else:
                ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            # getting the chosen activation function
            fn = self._get_activation(name)

            # activation function on hidden
            h = fn(h)

            # computing the new state as the sum of previous
            # state and the output of the current operation (see paper definition at page 2)
            s = s_prev + c * (h - s_prev)
            states += [s]

        # computing the output as the mean of the output of
        # the INTERMEDIATE nodes, where their index are
        # defined by the "concat" list in the genotype
        output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
        return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 ntoken,
                 ninp,
                 nhid,
                 nhidlast,
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1,
                 num_intermediate_nodes=8,
                 handle_hidden_mode=None,
                 cell_cls=DARTSCell,
                 genotype=None):

        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.handle_hidden_mode = handle_hidden_mode
        self.num_intermediate_nodes = num_intermediate_nodes
        self.encoder = nn.Embedding(ntoken, ninp)

        assert ninp == nhid == nhidlast
        if cell_cls == DARTSCell:
            assert genotype is not None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype, self.num_intermediate_nodes,
                                  handle_hidden_mode=handle_hidden_mode)]
        else:
            assert genotype is None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, num_intermediate_nodes=num_intermediate_nodes,
                                  handle_hidden_mode=handle_hidden_mode)]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, return_h=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
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

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [Variable(weight.new(1, bsz, self.nhid).zero_())]
