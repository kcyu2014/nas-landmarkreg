#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/26 下午4:13
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================


# search code here.
import torch, random
import torch.nn as nn
from torch.distributions import Categorical
from copy import deepcopy
import logging

from nasws.dataset.image_datasets import Dataset2Class
from .lib.models import get_search_spaces
from .lib.models.cell_operations import ResNetBasicblock
from .lib.models.cell_searchs.search_cells import SearchCell, SearchCellOnNode, SearchCellPCDarts
from .lib.models.cell_searchs.genotypes import Structure
from ..supernet import Supernet


class NASBench201NetSearch(Supernet):

    def __init__(self, init_channel, num_layers, max_nodes, num_classes, op_choices, args=None, cell_fn=None):
        """
        
        :param init_channel: init channel number 
        :param num_layers: 
        :param max_nodes: Fixed during search.
        :param num_classes: number classes
        :param op_choices: this indicates operation choices.
        """
        super(NASBench201NetSearch, self).__init__(args)
        self.args = args
        self._C = init_channel
        self._layerN = num_layers
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(init_channel, track_running_stats=args.wsbn_track_stat, affine=args.wsbn_affine))

        layer_channels = [init_channel] * num_layers + [init_channel * 2] +         [init_channel * 2] * num_layers + \
                         [init_channel * 4] + [init_channel * 4] * num_layers
        layer_reductions = [False] * num_layers + [True] + [False] * num_layers + [True] + [False] * num_layers
        logging.debug(f'NASbench 201 search cell channels {layer_channels}')
        logging.debug(f'NASbench 201 search reductions {layer_reductions}')
        
        C_prev, num_edge, edge2index = init_channel, None, None
        self.cells = nn.ModuleList()
        
        num_cells = len(layer_channels)
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, args)
            else:
                if cell_fn:
                    cell = cell_fn(C_prev, C_curr, 1, max_nodes, op_choices, args, index, num_cells)
                else:
                    if args.supernet_cell_type == 'op_on_edge':
                        cell = SearchCell(C_prev, C_curr, 1, max_nodes, op_choices, args, index, num_cells)
                    elif args.supernet_cell_type == 'op_on_node':
                        cell = SearchCellOnNode(C_prev, C_curr, 1, max_nodes, op_choices, args, index, num_cells)
                    else:
                        raise NotImplementedError('WRONG in creating NB201 model search')
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(
                        num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(op_choices)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.dropout = nn.Dropout(args.global_dropout_rate)
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, track_running_stats=args.wsbn_track_stat, affine=args.wsbn_affine), 
            nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_cache = None

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(Init-Channels={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(
            name=self.__class__.__name__,
            **self.__dict__))

    def forward_oneshot(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_dynamic(feature, self.arch_cache)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        
        # adding global pooling dropout
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits, out

    # override the spec to arch_cache for temp reasons.
    @property
    def model_spec_cache(self):
        return self.arch_cache

    @model_spec_cache.setter
    def model_spec(self, spec):
        self.arch_cache = spec


class NASBench201NetSearchDARTS(NASBench201NetSearch):

    def __init__(self, init_channel, num_layers, max_nodes, num_classes, op_choices, args=None, cell_fn=None) -> None:
        cell_fn = SearchCellPCDarts if 'pcdarts' in args.search_policy else cell_fn
        cell_fn = cell_fn or SearchCell
        logging.info(f'>>> Creating Differentiable NASBenc201Net : cell_fn {cell_fn}')
        super(NASBench201NetSearchDARTS, self).__init__(
            init_channel, num_layers, max_nodes, num_classes, op_choices, args, cell_fn
        )
        # initialize the other network
        # create Alpha here
        num_edge = self.cells[-1].num_edges
        self.arch_parameters = [1e-3 * torch.randn(num_edge, len(op_choices))]
    
    def forward_darts(self, inputs):
        alphas = nn.functional.softmax(self.arch_parameters[0], dim=-1)

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits, out
    
    def forward_gdas(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_gdas(feature, self.arch_parameters[0], self.args.gdas_tau)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling( out )
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits, out
    
    def _compute_softoneshot_alphas(self, model_spec):
        alphas = torch.ones_like(self.arch_parameters[0])
        alphas /= alphas.sum(dim=1, keepdim=True)
        # mutate based on self.arch_cache
        # spec.nodes = [((op, 0)), ((op, 0), (op, 1)), ((op, 0), (op, 1), (op, 2))]
        # just do a sequential order
        # given the current 
        delta = self.args.softoneshot_alpha
        n = len(self.op_names)
        for i in range(1, model_spec.node_num):
            cur_op_node = model_spec.nodes[i-1]
            for op_name, j in cur_op_node:
                edge_str = '{:}<-{:}'.format(i, j)
                op_index = self.op_names.index(op_name)
                edge_index = self.cells[0].edge2index[edge_str]
                alphas[edge_index] -= delta / (n - 1)
                alphas[edge_index, op_index] += n / (n-1) * delta
        return alphas

    def forward_softoneshot(self, inputs):
        alphas = self._compute_softoneshot_alphas(self.arch_cache)
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return logits, out

    def genotype(self, method='argmax'):
        """Sample the genotype

        Parameters
        ----------
        method : str, optional
            sampling method, by default 'argmax'
            'argmax' - use the max to set genotype, fix everytime
            'random' - use uniform random sampling, will change from time to time

        Returns
        -------
        [type]
            [description]
        """
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[0][self.edge2index[node_str]]
                    if method == 'argmax':
                        op_name = self.op_names[weights.argmax().item()]
                    elif method == 'random':
                        op_name = self.op_names[Categorical(logits=weights).sample()]
                    else:
                        raise NotImplementedError(f'genotype method {method} not supported')
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)


def build_nasbench201_search_model(args):
    """ build this with default configuration. Unless changed, this should not be changed. """
    op_choices = get_search_spaces('cell', 'aa-nas')
    # default argus: init_channels = 16 layers = 5
    if 'darts' in args.search_policy or 'gdas' in args.search_policy:
        net = NASBench201NetSearchDARTS(args.init_channels, args.layers, 
                                max_nodes=args.num_intermediate_nodes,
                                num_classes=Dataset2Class[args.dataset], op_choices=op_choices, args=args)
    else:
        net = NASBench201NetSearch(args.init_channels, args.layers, 
                                max_nodes=args.num_intermediate_nodes,
                                num_classes=Dataset2Class[args.dataset], op_choices=op_choices, args=args)
    return net
