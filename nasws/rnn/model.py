import torch
# after import torch, add this
import utils

torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils import mask2d
from utils import LockedDropout
from utils import embedded_dropout
from torch.autograd import Variable
import numpy as np

INITRANGE = 0.04
model_logger = logging.getLogger("train_search.model")


class BenchmarkCell(nn.Module):
    """
    > Inherit from DARTS cell.

    This cell is when performing node = 2 experiment.
        You should never modify the operation without testing!

    """

    def __init__(self, ninp, nhid, dropouth, dropoutx, genotype, use_glorot,
                 steps,
                 genotype_id,
                 args,
                 handle_hidden_mode=None):
        """
        :param ninp: input size (word embedding size)
        :param nhid: hidden state size (number of hidden units of the FC layer in a rnn cell)
        :param dropouth: dropout rate for hidden
        :param dropoutx: dropout rate for input
        :param genotype: string representation of the cell (description of the edges involved)
        """
        super(BenchmarkCell, self).__init__()
        self.nhid = nhid
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.genotype = genotype
        self.genotype_id = genotype_id
        self.args = args
        self.emulate_ws_solutions = self.args.emulate_ws_solutions
        self.handle_hidden_mode = handle_hidden_mode

        model_logger.info("CELL INITIALIZED ")

        # IMPORTANT: genotype is None when doing arch search
        # the steps are equal to the number of intermediate nodes
        # in the cell (default 4 nodes for cnns, 8 nodes for rnn)
        self.steps = len(self.genotype.recurrent) if self.genotype is not None else steps

        if self.emulate_ws_solutions is not None:

            self._W0s = nn.ParameterList([])
            self.w0s_dict = {}
            self._Ws_matrices = nn.ParameterList([])
            self.matrices_dict = {}

            for geno_id in range(self.args.genotype_start, self.args.genotype_end):
                self.w0s_dict[geno_id] = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))
                self.matrices_dict[geno_id] = [
            nn.Parameter(torch.Tensor(nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for _ in range(self.steps)
                                    ]

                self._W0s.append(self.w0s_dict[geno_id])
                self._Ws_matrices.extend(self.matrices_dict[geno_id])
                self.matrices_dict[geno_id] = nn.ParameterList(self.matrices_dict[geno_id])

            self._W0 = self.w0s_dict[self.args.genotype_start]
            self._Ws = self.matrices_dict[self.args.genotype_start]

        else:
            # initializing the first weight matrix between input x and the hidden layer
            self._W0 = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))

            # initializing the weight matrices used towards the intermediate nodes (range of steps)
            self._Ws = nn.ParameterList([
                nn.Parameter(torch.Tensor(nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for _ in range(self.steps)
                                        ])

            if use_glorot:
                for param in self._Ws:
                    nn.init.xavier_normal(param)

    @property
    def Ws(self):
        return self._Ws

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

        if self.emulate_ws_solutions is not None:
            self._W0 = self.w0s_dict[self.genotype_id]
            self._Ws = self.matrices_dict[self.genotype_id]
            # model_logger.info('weights matrices changed for the new geno_id: {}, genotype: {}'.format(self.genotype_id, self.genotype))

        # forward pass through time in the cell, T is the sequence length
        for t in range(T):

            # main forward inside the cell
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask)

            if self.handle_hidden_mode == 'NORM':
                hidden_norms = hidden.norm(dim=-1)
                max_norm = 25.0
                if hidden_norms.data.max() > max_norm:
                    # Just directly use the torch slice operations
                    # in PyTorch v0.4.
                    #
                    # This workaround for PyTorch v0.3.1 does everything in numpy,
                    # because the PyTorch slicing and slice assignment is too
                    # flaky.
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

            # saving the hidden output for each step
            hiddens.append(hidden)

        if clipped_num > 0:
            model_logger.debug('clipped {} hidden states in one forward '.format(clipped_num))
            model_logger.debug('max clipped hidden state norm: {}'.format(max_clipped_norm))

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
            f = utils.tanh
        elif name == 'relu':
            f = F.relu
        elif name == 'sigmoid':
            f = utils.sigmoid
        elif name == 'identity':
            f = lambda x: x
        elif name == 'selu':
            f = torch.nn.functional.selu
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
            edge_weights_id = i

            # applying dropout masks if training.
            # computing the matrix mul between the previous output
            # and the weights of the current node "i" (FC layer)
            if self.training:
                ch = (s_prev * h_mask).mm(self.Ws[edge_weights_id])
            else:
                ch = s_prev.mm(self.Ws[edge_weights_id])

            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            # getting the chosen activation function
            fn = self._get_activation(name)

            # activation function on hidden
            h = fn(h)

            s = s_prev + c * (h - s_prev)

            states += [s]

        # computing the output as the mean of the output of
        # the INTERMEDIATE nodes, where their index are
        # defined by the "concat" list in the genotype
        # print('CONCAT STATES USED {}:'.format(self.genotype.concat))
        output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)

        # avoid the explosion of the hidden state by forcing the value in [-1, 1]
        if self.handle_hidden_mode == 'ACTIVATION':
            output = F.tanh(output)
        return output

    def set_genotype(self, genotype, genotype_id):
        """
        setting a new genotype for the DAG
        :param genotype: new genotype to be used in the forward of the cell
        """
        self.genotype = genotype
        self.genotype_id = genotype_id


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 ntoken,
                 args,
                 genotype_id,
                 cell_cls=BenchmarkCell,
                 genotype=None):

        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, args.emsize)
        self.nlayers = args.nlayers
        self.rnns = []

        assert args.emsize == args.nhid == args.nhidlast

        for layers in range(args.nlayers):
            if cell_cls == BenchmarkCell or issubclass(cell_cls, BenchmarkCell):
                assert genotype is not None
                self.rnns.append(cell_cls(args.emsize,
                                          args.nhid,
                                          args.dropouth,
                                          args.dropoutx,
                                          genotype,
                                          args.use_glorot,
                                          args.num_intermediate_nodes,
                                          genotype_id,
                                          args,
                                          args.handle_hidden_mode)
                                 )
            else:
                assert genotype is None
                self.rnns = [cell_cls(args.emsize, args.nhid, args.dropouth, args.dropoutx)]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(args.emsize, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp = args.emsize
        self.nhid = args.nhid
        self.nhidlast = args.nhidlast
        self.dropout = args.dropout
        self.dropouti = args.dropouti
        self.dropoute = args.dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls
        self.args = args

        # model_logger.info("MODEL INITIALIZED ")

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, return_h=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)
        # logging.debug('embedding norm', emb.norm())

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            # print("raw output norm: ", raw_output.norm())
            # print("new hidden norm: ", new_h.norm())

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
        init_hidden_list = list()
        for layer in range(self.nlayers):
            init_h = Variable(weight.new(1, bsz, self.nhid).zero_())
            init_hidden_list.append(init_h)

        return init_hidden_list

    def change_genotype(self, genotype, genotype_id):

        for rnn in self.rnns:
            rnn.set_genotype(genotype=genotype, genotype_id=genotype_id)
        return self

    def genotype(self):
        return self.rnns[0].genotype

    def genotype_id(self):
        return self.rnns[0].genotype_id
