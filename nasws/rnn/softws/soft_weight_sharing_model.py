from functools import partial

import IPython
import torch
# after import torch, add this
import utils
from nasws.rnn.model import RNNModel, BenchmarkCell

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


def default_genotype_to_param_query(geno, geno_id, num_param_per_node, steps):
    return [[1.0 / num_param_per_node[j], ] * num_param_per_node[j] for j in range(steps)]


def get_fn_map(query, args, **kwargs):
    if query == 'default' or query is None:
        f = default_genotype_to_param_query
    elif query == 'soft_map_v3_init':
        logging.info("Soft Weight Sharing mapping v3: init_v = {}".format(args.softws_init_v))
        f = partial(map_fn.soft_map_v3_init, init_v=args.softws_init_v)
    else:
        f = getattr(map_fn, query)
    logging.info("Soft weight sharing mapping {}".format(query))

    if type(f) is not type:
        if callable(f):
            return f

    if not issubclass(f, map_fn.StoreQueryFull):
        return f
    else:
        # slightly complicated mapping function.
        f_map = f(**kwargs)
        if callable(f_map):
            return f_map
        else:
            raise ValueError("Get map fn wrong!")


class BenchmarkCellSoftWS(BenchmarkCell):
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
        super(BenchmarkCell, self).__init__()
        self.nhid = nhid
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.genotype = genotype
        self.genotype_id = genotype_id
        self.args = args
        self.emulate_ws_solutions = self.args.emulate_ws_solutions
        self.handle_hidden_mode = handle_hidden_mode

        # model_logger.info("CELL INITIALIZED ")

        # IMPORTANT: genotype is None when doing arch search
        # the steps are equal to the number of intermediate nodes
        # in the cell (default 4 nodes for cnns, 8 nodes for rnn)
        self.steps = len(self.genotype.recurrent) if self.genotype is not None else steps
        self.num_param_per_node = [self.args.softws_num_param_per_node, ] * self.steps

        # Do not share the input layer for now.
        # initializing the first weight matrix between input x and the hidden layer
        _w0 = torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)
        if use_glorot:
            nn.init.xavier_normal_(_w0)
        self._W0 = nn.Parameter(_w0)

        # initializing the weight matrices used towards the intermediate nodes (range of steps)
        # num-steps initial values
        self.initialize_ws_by_values()
        self.genotype_to_param_query = get_fn_map(args.genotype_mapping_fn,
                                                  num_param_per_node=self.num_param_per_node,
                                                  steps=steps,
                                                  search_space=args.search_space,
                                                  args=args
                                                  )

    def _initialize_ws_values(self, nhid=None):
        nhid = nhid or self.nhid
        init_values = []
        for i in range(self.steps):
            i_p = torch.Tensor(nhid, 2 * nhid)
            if not self.args.use_glorot:
                i_p.uniform_(-INITRANGE, INITRANGE)
            else:
                nn.init.xavier_normal_(i_p)
            init_values.append(i_p)
        return init_values

    def initialize_ws_by_values(self, init_tensors=None):
        # If no input tensor, just update accordingly.
        def _get_init_paramters_list(num_param, i_p):
            list_params = []
            for i in range(num_param):
                list_params.append(
                    nn.Parameter(i_p.detach().clone())
                )
            return list_params
        init_tensors = init_tensors or self._initialize_ws_values()
        assert len(init_tensors) == self.steps
        # delete the previous parameters
        self.soft_param_dicts = {}
        self._Ws = None
        _Ws = []
        for i in range(self.steps):
            self.soft_param_dicts[i] = []
            new_ws = _get_init_paramters_list(self.num_param_per_node[i], init_tensors[i])
            for j in range(self.num_param_per_node[i]):
                _Ws.append(new_ws[j])
                self.soft_param_dicts[i].append(new_ws[j])

        # Register the weights to nn.Module.
        self._Ws = nn.ParameterList(_Ws)

    def compute_ws(self, genotype=None, genotype_id=None):
        genotype = genotype or self.genotype
        genotype_id = genotype_id or self.genotype_id
        return self._compute_ws(
            self.genotype_to_param_query(
                genotype, genotype_id, self.num_param_per_node, self.steps))

    def _compute_ws(self, matrix_query):
        """
        For a set of weight matrix, obtain the _Ws, from matrix query.
        :param matrix_query:
        :return:
        """
        ws = []
        for i in range(self.steps):
            w = 0
            for ind, j in enumerate(matrix_query[i]):
                w = w + j * self.soft_param_dicts[i][ind]
            ws.append(w)
        return ws

    @property
    def Ws(self):
        return self.compute_ws()


class RNNModelSoftWS(RNNModel):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 ntoken,
                 args,
                 genotype_id,
                 cell_cls=BenchmarkCellSoftWS,
                 genotype=None,
                 ):

        super(RNNModelSoftWS, self).__init__(ntoken,
                 args,
                 genotype_id,
                 cell_cls,
                 genotype)

################################
######## Experimental ########
################################

