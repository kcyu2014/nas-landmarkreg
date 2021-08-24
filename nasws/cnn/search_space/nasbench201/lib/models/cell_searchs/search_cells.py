import logging
import math
import random
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..cell_operations import OPS, WSBN_OPS
from nasws.cnn.operations.dropout import apply_drop_path
import IPython


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class SearchCell(nn.Module):
    """
    WSBN does not apply to this search cell. because
    each edge has complete ops initialized
    i.e.  for this edge, input is always coming from the same node!
      node i - > node j: OPS
    """

    def __init__(self, C_in, C_out, stride, max_nodes, op_names,
                 args=None, layer_idx=None, max_layer_num=None):
        super(SearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out

        self.args = args
        self.layer_idx = layer_idx
        self.max_layer_num = max_layer_num
        bn_kwargs = {'track_running_stats': args.wsbn_track_stat,
                     'affine': args.wsbn_affine} if args else {}
        logging.debug("Setting BN-kwargs to {}".format(bn_kwargs))
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    xlists = [WSBN_OPS[op_name](
                        C_in, C_out, stride, bn_kwargs) for op_name in op_names]
                else:
                    xlists = [WSBN_OPS[op_name](
                        C_in, C_out,      1, bn_kwargs) for op_name in op_names]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(
            **self.__dict__)
        return string

    # DARTS forward
    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    # as MixedOp
                    sum(layer(nodes[j]) * w for layer,
                        w in zip(self.edges[node_str], weights))
                )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # GDAS
    def forward_gdas(self, inputs, alphas, _tau):
        avoid_zero = 0
        while True:
            gumbels = -torch.empty_like(alphas).exponential_().log()
            logits = (alphas.log_softmax(dim=1) + gumbels) / _tau
            probs = nn.functional.softmax(logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                continue  # avoid the numerical error
            nodes = [inputs]

            for i in range(1, self.max_nodes):
                inter_nodes = []
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    weights = hardwts[self.edge2index[node_str]]
                    argmaxs = index[self.edge2index[node_str]].item()
                    # make sure the gradient is passed correctly.
                    # so this is always SPOS training but using gradient to do the BP.
                    weigsum = sum(weights[_ie] * edge(nodes[j]) if _ie == argmaxs else weights[_ie]
                                  for _ie, edge in enumerate(self.edges[node_str]))
                    inter_nodes.append(weigsum)
                nodes.append(sum(inter_nodes))
            avoid_zero += 1
            if nodes[-1].sum().item() == 0:
                if avoid_zero < 10:
                    continue
                else:
                    warnings.warn(
                        'get zero outputs with avoid_zero={:}'.format(avoid_zero))
                    break
            else:
                break
        return nodes[-1]

    # joint
    def forward_joint(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                # aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
                aggregation = sum(
                    layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights))
                inter_nodes.append(aggregation)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # uniform random sampling per iteration
    def forward_urs(self, inputs):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            while True:  # to avoid select zero for all ops
                sops, has_non_zero = [], False
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    candidates = self.edges[node_str]
                    select_op = random.choice(candidates)
                    sops.append(select_op)
                    if not hasattr(select_op, 'is_zero') or select_op.is_zero == False:
                        has_non_zero = True
                if has_non_zero:
                    break
            inter_nodes = []
            for j, select_op in enumerate(sops):
                inter_nodes.append(select_op(nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # select the argmax
    def forward_select(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    self.edges[node_str][weights.argmax().item()](nodes[j]))
                #inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # forward with a specific structure
    def forward_dynamic(self, inputs, structure):

        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i-1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = '{:}<-{:}'.format(i, j)
                op_index = self.op_names.index(op_name)
                inter_nodes.append(self.edges[node_str][op_index](nodes[j]))
            nodes.append(sum(inter_nodes))

        out = nodes[-1]
        if self.args.path_dropout_rate > 0:
            out = apply_drop_path(
                out, 1 - self.args.path_dropout_rate, self.layer_idx, self.max_layer_num, 0, 1)
        return out

    # forward with a specific structure with WSBN op
    def forward_dynamic_wsbn(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i-1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = '{:}<-{:}'.format(i, j)
                op_index = self.op_names.index(op_name)
                inter_nodes.append(self.edges[node_str][op_index](nodes[j], j))
            nodes.append(sum(inter_nodes))

        out = nodes[-1]
        if self.args.path_dropout_rate > 0:
            out = apply_drop_path(
                out, 1 - self.args.path_dropout_rate, self.layer_idx, self.max_layer_num, 0, 1)
        return out


class SearchCellOnNode(SearchCell):

    def __init__(self, C_in, C_out, stride, max_nodes, op_names,
                 args=None, layer_idx=None, max_layer_num=None):
        super(SearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        # store the real ops on the node
        self.nodes = nn.ModuleDict()
        self.edges = {}  # map the previous, but not as monitored weights.
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out

        self.args = args
        self.layer_idx = layer_idx
        self.max_layer_num = max_layer_num
        bn_kwargs = {'track_running_stats': args.wsbn_track_stat,
                     'affine': args.wsbn_affine} if args else {}
        logging.debug("Setting BN-kwargs to {}".format(bn_kwargs))
        # make sure these are resued? So the idea is, edges should be

        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_str = '{:}-{:}'.format(i, j)
                if j == 0:  # reduce node
                    xlists = [WSBN_OPS[op_name](
                        C_in, C_out, stride, bn_kwargs) for op_name in op_names]
                elif j == 1:
                    xlists = [WSBN_OPS[op_name](
                        C_in, C_out,      1, bn_kwargs) for op_name in op_names]
                else:  # other node.
                    # xlists = self.nodes['{:}-{:}'.format(i, 1)] # always use the j==1 node
                    self.edges[node_str] = '{:}-{:}'.format(i, 1)
                    continue
                # create the real list.
                self.nodes[op_str] = nn.ModuleList(xlists)
                self.edges[node_str] = op_str
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    # forward with a specific structure
    def forward_dynamic(self, inputs, structure):

        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i-1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = '{:}<-{:}'.format(i, j)
                op_index = self.op_names.index(op_name)
                op = self.nodes[self.edges[node_str]]
                inter_nodes.append(op[op_index](nodes[j]))
            nodes.append(sum(inter_nodes))

        out = nodes[-1]
        if self.args.path_dropout_rate > 0:
            out = apply_drop_path(
                out, 1 - self.args.path_dropout_rate, self.layer_idx, self.max_layer_num, 0, 1)
        return out

    # forward with a specific structure
    def forward_dynamic_wsbn(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i-1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = '{:}<-{:}'.format(i, j)
                op_index = self.op_names.index(op_name)
                op = self.nodes[self.edges[node_str]]
                inter_nodes.append(op[op_index](nodes[j], j))
            nodes.append(sum(inter_nodes))

        out = nodes[-1]
        if self.args.path_dropout_rate > 0:
            out = apply_drop_path(
                out, 1 - self.args.path_dropout_rate, self.layer_idx, self.max_layer_num, 0, 1)
        return out


class SearchCellPCDarts(SearchCell):
    """PCDarts Search Cell

    Parameters
    ----------
    SearchCell : [type]
        [description]
    """

    def __init__(self, C_in, C_out, stride, max_nodes, op_names,
                 args=None, layer_idx=None, max_layer_num=None):
        super(SearchCell, self).__init__()
        self.k = 4
        self.mp = nn.MaxPool2d(2,2)
        
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out

        self.args = args
        self.layer_idx = layer_idx
        self.max_layer_num = max_layer_num
        bn_kwargs = {'track_running_stats': args.wsbn_track_stat,
                     'affine': args.wsbn_affine} if args else {}
        logging.debug("Setting BN-kwargs to {}".format(bn_kwargs))
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                _stride = 1 if j == 0 else stride
                xlists = [WSBN_OPS[op_name](C_in // self.k, C_out // self.k, _stride, bn_kwargs) 
                            for op_name in op_names]
                # no batchnorm after this layer, so no need to judge
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)
        
    
    def pcdarts_op_forward(self, x, weights, ops):
        """PCDARTS forward function of each edge """
        # channel proportion k=4
        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2//self.k, :, :]
        xtemp2 = x[:,  dim_2//self.k:, :, :]
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, ops))
        # reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1, xtemp2], dim=1)
        else:
            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, self.k)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        # except channe shuffle, channel shift also works
        return ans

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            # i = steps
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                # New mixed op for pc-darts with channel sampling

                weights = weightss[self.edge2index[node_str]]
                x = nodes[j]
                ans = self.pcdarts_op_forward(x, weights, self.edges[node_str])
                inter_nodes.append(ans)

            state = sum(inter_nodes)
            nodes.append(state)

        return nodes[-1]

    def forward_dynamic(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i-1]
            inter_nodes = []
            # for j in range(i):
            for op_name, j in cur_op_node:
                # op_name = 
                node_str = '{:}<-{:}'.format(i, j)
                op_index = self.op_names.index(op_name)
                
                weights = F.one_hot(torch.tensor([op_index]), num_classes=len(self.op_names))[0].float().cuda()
                ans = self.pcdarts_op_forward(nodes[j], weights, self.edges[node_str])
                inter_nodes.append(ans)
            nodes.append(sum(inter_nodes))

        out = nodes[-1]
        if self.args.path_dropout_rate > 0:
            out = apply_drop_path(
                out, 1 - self.args.path_dropout_rate, self.layer_idx, self.max_layer_num, 0, 1)
        return out
