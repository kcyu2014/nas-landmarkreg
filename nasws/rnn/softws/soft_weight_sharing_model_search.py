import IPython
import torch
# after import torch, add this
import utils
from .soft_weight_sharing_model import BenchmarkCellSoftWS, RNNModelSoftWS

torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import logging
from utils import mask2d
from utils import LockedDropout
from utils import embedded_dropout
from torch.autograd import Variable
import numpy as np

import nasws.rnn.softws.soft_ws_mapping_node2 as map_fn


INITRANGE = 0.04
model_logger = logging.getLogger("train_search.model")


class BenchmarkCellSoftWSSearch(BenchmarkCellSoftWS):
    """
    > Inherit from DARTS cell.
    A first version of hard sharing system. Test later.

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

        super(BenchmarkCellSoftWSSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype, use_glorot,
                 steps,
                 genotype_id,
                 args,
                 handle_hidden_mode)
        self.bn = nn.BatchNorm1d(nhid, affine=False)
        self.handle_hidde_mode = handle_hidden_mode

    def compute_ws(self, genotype=None, genotype_id=None):
        genotype = genotype or self.genotype
        genotype_id = genotype_id or self.genotype_id
        # query = self.genotype_to_param_query(
        #         genotype, genotype_id, self.num_param_per_node, self.steps)
        # TODO one day change this to cluster center id.
        query = self._softws_parameters[genotype_id]

        # Make a softmax out of it.
        probs = F.softmax(query, dim=-1)
        # IPython.embed(header="Debug compute_ws")
        return self._compute_ws(probs)

    def _compute_ws(self, matrix_query):
        """
        For a set of weight matrix, obtain the _Ws, from matrix query.
        :param matrix_query:
        :return:
        """
        ws = []
        for i in range(self.steps):
            w = 0
            # IPython.embed(header=' Debug and test if i could remove the for loop')
            for ind, param in enumerate(self.soft_param_dicts[i]):
                w = w + param * matrix_query[i][ind]
            ws.append(w)
        return ws


class RNNModelSoftWSSearch(RNNModelSoftWS):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                    *args,
                    **kwargs
                 ):

        super(RNNModelSoftWSSearch, self).__init__(*args, cell_cls=BenchmarkCellSoftWSSearch, **kwargs)
        self._args = args
        self._init_softws_parameters()

    def _init_softws_parameters(self):
        # create the geno-id to weights,
        self._softws_parameters = []

        assert hasattr(self.rnns[0].genotype_to_param_query, 'id_query'), \
            'genotype_to_param_query must has id_query dict!' \
            'Make sure you use "StoreFull" in soft_ws_mapping_node2.py '
        id_query = self.rnns[0].genotype_to_param_query.id_query
        for i in range(self.args.search_space.num_solutions):
            # do not register as parameter, because Parameter may register itself.
            # just use tensor with require grads.
            tmp_w = torch.from_numpy(np.asanyarray(id_query[i], dtype=np.float32))
            # tmp_w.cuda()
            tmp_w.requires_grad_(True)
            self._softws_parameters.append(tmp_w)

        for rnn in self.rnns:
            rnn._softws_parameters = self._softws_parameters

    def softws_parameters(self):
        # Do not register this with model_parameters.
        return self._softws_parameters

    def new(self):
        model_new = RNNModelSoftWSSearch(*self._args, genotype=self.genotype())
        for x, y in zip(model_new.softws_parameters(), self.softws_parameters()):
            x.data.copy_(y.detach().clone())
        model_new.change_genotype(self.genotype(), self.genotype_id())
        if self.args.cuda:
            if self.args.single_gpu:
                parallel_model = model_new.cuda()
            else:
                parallel_model = nn.DataParallel(model_new, dim=1).cuda()
        else:
            parallel_model = model_new
        return parallel_model

    def _loss(self, hidden, input, target):
        log_prob, hidden_next = self.forward(input, hidden, return_h=False)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
        return loss, hidden_next

    def _apply(self, fn):
        # Update the paramters should be like this!
        for param in self._softws_parameters:
            param.data = fn(param.data)
            if param.grad is not None:
                param.grad = fn(param.grad)

        for rnn in self.rnns:
            rnn._softws_parameters = self._softws_parameters

        return super(RNNModelSoftWSSearch, self)._apply(fn)

