import torch
import torch.nn as nn

# from nasws.cnn.policy.darts_policy.utils import drop_path
from nasws.cnn.operations.dropout import apply_drop_path

from .operations import DartsSpaceMixedVertex, Identity, DartsSpaceMixedVertexWSBN, DartsSpaceMixedVertexOFA
from .model import DartsSpaceCell


class DartsSpaceCellSearch(DartsSpaceCell):

    def __init__(self, genotype, channel_prev_prev, channel_prev, channel, reduction, reduction_prev, args=None, layer_idx=None, max_num_layer=None):
        super(DartsSpaceCellSearch, self).__init__(genotype, channel_prev_prev, channel_prev, channel, reduction, reduction_prev, args=args)
        self.args = args
        self.layer_idx = layer_idx
        self.max_num_layer = max_num_layer
        self._compile(channel, None, None, None, reduction)

    def _compile(self, channel, op_names, indices, concat, reduction):

        self._steps = self.args.num_intermediate_nodes
        self._concat = list(range(2, 2+self.args.num_intermediate_nodes))
        self.multiplier = self._steps
        self._indices = None

        self._ops = nn.ModuleList()
        for index in range(self._steps * 2): # create 2 times
            if self.args.nasbenchnet_vertex_type == 'mixedvertex':
                if self.args.bn_type == 'bn':
                    op = DartsSpaceMixedVertex(channel, channel, 'sep_conv_3x3', reduction, args=self.args, curr_vtx_id=index)
                elif self.args.bn_type == 'wsbn':
                    op = DartsSpaceMixedVertexWSBN(channel, channel, 'sep_conv_3x3', reduction, args=self.args, curr_vtx_id=index)
            elif self.args.nasbenchnet_vertex_type == 'mixedvertex_ofa':
                op = DartsSpaceMixedVertexOFA(channel, channel, 'sep_conv_3x3', reduction, args=self.args, curr_vtx_id=index)
            elif self.args.nasbenchnet_vertex_type == 'mixedvertex_edge':
                op = None
            self._ops += [op]

    def change_genotype(self, genotype):
        if self.reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._concat = concat
        self.multiplier = len(concat)
        self._indices = indices

        for ind, (op, idx) in enumerate(zip(op_names, indices)):
            self._ops[ind].change_vertex_type(op, proj_op_ids=[idx])   # add this to differentiate the projection ops

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        try:
            states = [s0, s1]
            curr_step = int(len(self._indices) / 2)
            # IPython.embed()
            for i in range(curr_step):
                h1 = states[self._indices[2 * i]]
                h2 = states[self._indices[2 * i + 1]]
                op1 = self._ops[2 * i]
                op2 = self._ops[2 * i + 1]
                h1 = op1(h1)
                h2 = op2(h2)
                if self.training and drop_prob > 0.:
                    if not isinstance(op1, Identity):
                        h1 = apply_drop_path(h1, drop_prob, self.layer_idx, self.max_num_layer, i, self._steps)
                    if not isinstance(op2, Identity):
                        h2 = apply_drop_path(h2, drop_prob, self.layer_idx, self.max_num_layer, i, self._steps)
                s = h1 + h2
                states += [s]
        except Exception as e:
            print(e)
            IPython.embed()
            raise e

        # cutting the gradients and force it to be zeros.
        zero_tensor = states[2].detach().mul(0.)

        # make sure this is zeroed if concat does not choose.
        return torch.cat([states[i] if i in self._concat else zero_tensor.clone()
                          for i in range(2,2+self._steps)], dim=1)


class DartsSpaceCellOpEdgeSearch(DartsSpaceCell):

    def __init__(self, genotype, channel_prev_prev, channel_prev, channel, reduction, reduction_prev, args=None, layer_idx=None, max_num_layer=None):
        super(DartsSpaceCellOpEdgeSearch, self).__init__(genotype, channel_prev_prev, channel_prev, channel, reduction, reduction_prev)
        self.args = args
        self.layer_idx = layer_idx
        self.max_num_layer = max_num_layer
        self._compile(channel, None, None, None, reduction)

    def _compile(self, channel, op_names, indices, concat, reduction):

        self._steps = self.args.num_intermediate_nodes
        self._concat = list(range(2, 2+self.args.num_intermediate_nodes))
        self.multiplier = self._steps
        self._indices = None

        self._ops = nn.ModuleDict()
        for ind in range(2, self._steps + 2):
            for jnd in range(ind):
                if self.args.nasbenchnet_vertex_type == 'mixedvertex':
                    if self.args.bn_type == 'bn':
                        op = DartsSpaceMixedVertex(channel, channel, 'sep_conv_3x3', reduction, args=self.args, curr_vtx_id=ind)
                    elif self.args.bn_type == 'wsbn':
                        op = DartsSpaceMixedVertexWSBN(channel, channel, 'sep_conv_3x3', reduction, args=self.args, curr_vtx_id=ind)
                elif self.args.nasbenchnet_vertex_type == 'mixedvertex_ofa':
                    op = DartsSpaceMixedVertexOFA(channel, channel, 'sep_conv_3x3', reduction, args=self.args, curr_vtx_id=ind)
                self._ops['{:}<-{:}'.format(jnd, ind)] = op
        
        self.edge_keys = sorted(list(self._ops.keys()))
        self.num_edges  = len(self._ops)
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}

    def change_genotype(self, genotype):
        if self.reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._concat = concat
        self.multiplier = len(concat)
        self._indices = indices

        for ind, (op, idx) in enumerate(zip(op_names, indices)):
            curr_node_id = ind // 2 + 2
            # add this to differentiate the projection ops
            self._ops["{:}<-{:}".format(idx, curr_node_id)].change_vertex_type(op, proj_op_ids=[idx])

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        try:
            states = [s0, s1]
            curr_step = int(len(self._indices) / 2)
            # IPython.embed()
            for i in range(curr_step):
                in1 = self._indices[2 * i]
                in2 = self._indices[2 * i + 1]
                curr_node_id = i + 2
                h1 = states[in1]
                h2 = states[in2]
                op1 = self._ops['{:}<-{:}'.format(in1, curr_node_id)]
                op2 = self._ops['{:}<-{:}'.format(in2, curr_node_id)]
                h1 = op1(h1)
                h2 = op2(h2)
                if self.training and drop_prob > 0.:
                    if not isinstance(op1, Identity):
                        h1 = apply_drop_path(h1, drop_prob, self.layer_idx, self.max_num_layer, i, self._steps)
                    if not isinstance(op2, Identity):
                        h2 = apply_drop_path(h2, drop_prob, self.layer_idx, self.max_num_layer, i, self._steps)
                s = h1 + h2
                states += [s]
        except Exception as e:
            print(e)
            raise e

        # cutting the gradients and force it to be zeros.
        zero_tensor = states[2].detach().mul(0.)

        # make sure this is zeroed if concat does not choose.
        return torch.cat([states[i] if i in self._concat else zero_tensor.clone()
                          for i in range(2,2+self._steps)], dim=1)


class DartsSpaceCellSearchV2(DartsSpaceCell):
    """Fix the edge weigths, created a nested sharing weights compare to the baseline.
    Specifically, 


    Parameters
    ----------
    DartsSpaceCell : [type]
        [description]
    """
    def __init__(self, genotype, channel_prev_prev, channel_prev, channel, reduction, reduction_prev, args=None, layer_idx=None, max_num_layer=None):
        super(DartsSpaceCellSearchV2, self).__init__(genotype, channel_prev_prev, channel_prev, channel, reduction, reduction_prev, args=args)
        self.args = args
        self.layer_idx = layer_idx
        self.max_num_layer = max_num_layer
        self._compile(channel, None, None, None, reduction)
        self._spec_cache = None

    def _compile(self, channel, op_names, indices, concat, reduction):

        self._steps = self.args.num_intermediate_nodes
        self._concat = list(range(2, 2+self.args.num_intermediate_nodes))
        self.multiplier = self._steps
        self._indices = None

        self._ops = nn.ModuleList()
        # Found a bug (or not really, a bug...)
        # design issue
        # total ops : 2 + 3 + ... steps + 2
        # e.g. when Node = 4, total ops = 14
        for index in range(self._steps): # create op per level
            for j in range(2 + index):
                """
                Note: This differs from the original Cell
                    Here the op is one of reduction or normal MixedOp
                    Since we do not need to share at all.
                """
                _reduce = True if reduction and j < 2 else False
                if self.args.nasbenchnet_vertex_type == 'mixedvertex':
                    if self.args.bn_type == 'bn':
                        op = DartsSpaceMixedVertex(channel, channel, 'sep_conv_3x3', _reduce, args=self.args, curr_vtx_id=index, only_reduce=True)
                    elif self.args.bn_type == 'wsbn':
                        op = DartsSpaceMixedVertexWSBN(channel, channel, 'sep_conv_3x3', _reduce, args=self.args, curr_vtx_id=index)
                elif self.args.nasbenchnet_vertex_type == 'mixedvertex_ofa':
                    op = DartsSpaceMixedVertexOFA(channel, channel, 'sep_conv_3x3', _reduce, args=self.args, curr_vtx_id=index)
                elif self.args.nasbenchnet_vertex_type == 'mixedvertex_edge':
                    op = None
                self._ops += [op]

    def change_genotype(self, genotype):
        if self.reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat

        self._concat = concat
        self.multiplier = len(concat)
        self._indices = indices
        offset = 0
        
        for i in range(self._steps):
            for j in range(2):
                op, idx = op_names[i*2 + j], indices[i*2 + j]
                self._ops[offset + idx].change_vertex_type(op, proj_op_ids=[idx])
            offset += i + 2
        self._spec_cache = genotype

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        try:
            states = [s0, s1]
            curr_step = int(len(self._indices) / 2)
            offset = 0

            for i in range(curr_step):
                prev_ids = self._indices[2 * i], self._indices[2 * i + 1]
                h1 = states[prev_ids[0]]
                h2 = states[prev_ids[1]]
                # op choice is different...
                op1 = self._ops[prev_ids[0] + offset]
                op2 = self._ops[prev_ids[1] + offset]
                # op1 = self._ops[2 * i]
                # op2 = self._ops[2 * i + 1]
                h1 = op1.forward_oneshot(h1)
                h2 = op2.forward_oneshot(h2)
                if self.training and drop_prob > 0.:
                    if not isinstance(op1, Identity):
                        h1 = apply_drop_path(h1, drop_prob, self.layer_idx, self.max_num_layer, i, self._steps)
                    if not isinstance(op2, Identity):
                        h2 = apply_drop_path(h2, drop_prob, self.layer_idx, self.max_num_layer, i, self._steps)
                s = h1 + h2
                offset += len(states)
                states += [s]
        except Exception as e:
            print(e)
            raise e

        # cutting the gradients and force it to be zeros.
        zero_tensor = states[2].detach().mul(0.)

        # make sure this is zeroed if concat does not choose.
        return torch.cat([states[i] if i in self._concat else zero_tensor.clone()
                          for i in range(2,2+self._steps)], dim=1)

    def forward_darts(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            try:
                # s = sum(self._ops[offset+j].forward_darts(h, weights[offset+j]) for j, h in enumerate(states))
                ress = []
                for j, h in enumerate(states):
                    ress.append(self._ops[offset+j].forward_darts(h, weights[offset+j]))
                s = sum(ress)
            except Exception as e:
                print(i)
                print([r.size() for r in ress])
                raise e

            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._steps:], dim=1)
    
    def forward_pcdarts(self, s0, s1, weights, weights2):
        # pcdarts forward is selected at Op level...
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset+j]*self._ops[offset+j].forward_pcdarts(h, weights[offset+j]) for j, h in enumerate(states))
            #s = channel_shuffle(s,4)
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self._steps:], dim=1)

    
    def forward_gdas(self, s0, s1, weightss, indexs):
        
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                op = self._ops[offset + j]
                weights = weightss[ offset + j ]
                index   = indexs[ offset + j ].item()
                clist.append(op.forward_gdas(h, weights, index))
            offset += len(states)
            states.append(sum(clist))
        
        return torch.cat(states[-self.multiplier:], dim=1)
