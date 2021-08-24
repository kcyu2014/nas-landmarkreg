import logging

from nasws.cnn.policy.darts_policy.operations import *
from nasws.cnn.policy.darts_policy.genotypes import PRIMITIVES
from ..api import MixedVertex


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


def random_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    indices = torch.randperm(num_channels)
    x = x[:,indices]
    return x

class OriginalDartsMixedVertex(nn.Module):

    # quite simple
    def __init__(self, channel, stride, vertex_type, args=None):
        super(OriginalDartsMixedVertex, self).__init__()
        self.args = args
        self._ops = nn.ModuleList()
        for p in PRIMITIVES:
            op = OPS[p](channel, stride, False)
            if 'pool' in p:
                op = nn.Sequential(op, nn.BatchNorm2d(channel, affine=False))
            self._ops.append(op)

        assert vertex_type in PRIMITIVES


class DartsSpaceMixedVertex(MixedVertex):

    def __init__(self, input_size, output_size, vertex_type, reduction, args=None,
                 curr_vtx_id=None, curr_cell_id=None, only_reduce=False):
        super(DartsSpaceMixedVertex, self).__init__(input_size, output_size, vertex_type,
                                                    args=args, curr_vtx_id=curr_vtx_id)
        self.reduce_ops = None
        self.reduction = reduction
        self.should_reduce = False
        self.mp = nn.MaxPool2d(2, 2)

        bn_kwargs = {'track_running_stats': args.wsbn_track_stat, 'affine': args.wsbn_affine} if args else {}
        logging.debug("Setting BN-kwargs to {}".format(bn_kwargs))
        if 'pcdarts' in args.search_policy:
            output_size = output_size // 4
            self.mode = 'pcdarts'
        
        elif 'darts' in args.search_policy:
            self.mode = 'darts'
        else:
            self.mode = 'oneshot'

        logging.debug("DartsMixedVertex mode {}".format(self.mode))

        if reduction:
            # create a separate list of reduced cells.
            ops = {}
            for p in PRIMITIVES:
                op = BN_OPS[p](output_size, 2, bn_kwargs)
                if 'pool' in p:
                    op = nn.Sequential(op, nn.BatchNorm2d(output_size, affine=False))
                ops[p] = op
            self.reduce_ops = nn.ModuleDict(ops)
            self.reduction = True
            if only_reduce:
                # do not create the self.ops. This serves as a sanity check
                return

        ops = {}
        for p in PRIMITIVES:
            op = BN_OPS[p](output_size, 1, bn_kwargs)
            if 'pool' in p:
                op = nn.Sequential(op, nn.BatchNorm2d(output_size, affine=False))
            ops[p] = op
        self.ops = nn.ModuleDict(ops)

    def change_vertex_type(self, vertex_type, proj_op_ids=None):
        """ proj_op_ids is a single element list, that contains the input node id. """
        self.vertex_type = vertex_type
        self.proj_op_ids = proj_op_ids 
        if any([i < 2 for i in proj_op_ids]):
            self.should_reduce = True
        else:
            self.should_reduce = False

    @property
    def current_op(self):
        if self.reduction and self.should_reduce:
            return self.reduce_ops[self.vertex_type]
        else:
            return self.ops[self.vertex_type]

    def forward(self, x, weight=None):
        if self.args.darts_search_shuffle == 'random':
            x = random_shuffle(x)
        elif self.args.darts_search_shuffle == 'channel':
            x = channel_shuffle(x, 2)

        if self.mode == 'oneshot':
            return self.forward_oneshot(x)
        elif self.mode == 'darts':
            return self.forward_darts(x, weight)
        elif self.mode == 'pcdarts':
            return self.forward_pcdarts(x, weight)
        
    def forward_oneshot(self, x):
        """
        Function it runs is
               relu(bn(w * \sum_i x_i + b))
        :param x:
        :param weight:
        :return:
        """
        return self.current_op(x)
        
    def forward_darts(self, x, weights):
        if self.reduction:
            return sum(w * self.reduce_ops[op](x) for w, op in zip(weights, PRIMITIVES))
        else:
            return sum(w * self.ops[op](x) for w, op in zip(weights, PRIMITIVES))
    
    def forward_gdas(self, x, weights, index):
        if self.reduction:
            return self.reduce_ops[PRIMITIVES[index]](x) * weights[index]
        else:
            return self.ops[PRIMITIVES[index]](x) * weights[index]

    def forward_pcdarts(self, x, weights):
        random_shuffle(x)
        dim_2 = x.shape[1]
        xtemp = x[ : , :  dim_2//4, :, :]
        xtemp2 = x[ : ,  dim_2//4:, :, :]
        if self.reduction:
            temp1 = sum(w * self.reduce_ops[op](xtemp) for w, op in zip(weights, PRIMITIVES))
        else:
            temp1 = sum(w * self.ops[op](xtemp) for w, op in zip(weights, PRIMITIVES))
        if temp1.shape[2] == x.shape[2]:
            ans = torch.cat([temp1,xtemp2],dim=1)
        else:
            ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
        return ans
 

class DartsSpaceMixedVertexWSBN(DartsSpaceMixedVertex):

    def __init__(self, input_size, output_size, vertex_type, reduction, args=None, curr_vtx_id=None, curr_cell_id=None):
        super(DartsSpaceMixedVertex, self).__init__(input_size, output_size, vertex_type,
                                                    args=args, curr_vtx_id=curr_vtx_id)
        self.reduce_ops = None
        self.reduction = reduction
        self.should_reduce = False

        if reduction:
            # create a separate list of reduced cells.
            ops = {}
            for p in PRIMITIVES:
                op = WSBNOPS[p](output_size, 2, False, curr_vtx_id + 2)
                if 'pool' in p:
                    op = nn.Sequential(op, nn.BatchNorm2d(output_size, affine=False,  track_running_stats=False))
                ops[p] = op
            self.reduce_ops = nn.ModuleDict(ops)
            self.reduction = True
        ops = {}
        for p in PRIMITIVES:
            op = WSBNOPS[p](output_size, 1, False, curr_vtx_id + 2)
            if 'pool' in p:
                op = nn.Sequential(op, nn.BatchNorm2d(output_size, affine=False, track_running_stats=False))
            ops[p] = op
        self.ops = nn.ModuleDict(ops)

    def forward(self, x, weight=None):
        """
        Function it runs is
               relu(bn(w * \sum_i x_i + b))
        :param x:
        :param weight:
        :return:
        """
        # always set this node to Train mode, for batch normalization.
        # self.train()
        if weight is not None:
            logging.warning("not yet support reduce_ops, todo later.")
            return sum(w * self.ops[op](x) for w, op in zip(weight, PRIMITIVES))
        else:
            return self.current_op.forward_wsbn(x, self.proj_op_ids[0]) if 'pool' not in self.vertex_type else self.current_op(x)



class DartsSpaceMixedVertexOFA(DartsSpaceMixedVertex):
                
    def __init__(self, input_size, output_size, vertex_type, reduction, args=None,
                 curr_vtx_id=None, curr_cell_id=None):
        super(DartsSpaceMixedVertex, self).__init__(input_size, output_size, vertex_type,
                                                    args=args, curr_vtx_id=curr_vtx_id)
        self.reduce_ops = None
        self.reduction = reduction
        self.should_reduce = False
        bn_kwargs = {'track_running_stats': args.wsbn_track_stat, 'affine': args.wsbn_affine} if args else {}
        logging.debug("Setting BN-kwargs to {}".format(bn_kwargs))
      
        if reduction:
            # create a separate list of reduced cells.
            ops = {}
            for p in PRIMITIVES:
                p = 'sep_conv' if p.startswith('sep') else p
                p = 'dil_conv' if p.startswith('dil') else p
                op = OFA_OPS[p](output_size, 2, bn_kwargs)
                if 'pool' in p:
                    op = nn.Sequential(op, nn.BatchNorm2d(output_size, affine=False))
                ops[p] = op
            self.reduce_ops = nn.ModuleDict(ops)
            self.reduction = True
        ops = {}
        for p in PRIMITIVES:
            p = 'sep_conv' if p.startswith('sep') else p
            p = 'dil_conv' if p.startswith('dil') else p
            op = OFA_OPS[p](output_size, 1, bn_kwargs)
            if 'conv' in p:
                op.change_kernel_size(3)
            if 'pool' in p:
                op = nn.Sequential(op, nn.BatchNorm2d(output_size, affine=False))
            ops[p] = op
        
        self.ops = nn.ModuleDict(ops)

    def change_vertex_type(self, vertex_type, proj_op_ids=None):
        """ proj_op_ids is a single element list, that contains the input node id. """
        if 'conv' in vertex_type:
            kernel_size = int(vertex_type[-1])
            self.vertex_type = vertex_type[:8]
            self.current_op.change_kernel_size(kernel_size)
        else:
            self.vertex_type = vertex_type
        
        self.proj_op_ids = proj_op_ids 
        if any([i < 2 for i in proj_op_ids]):
            self.should_reduce = True
        else:
            self.should_reduce = False
    

class DartsSpaceMixedVertexEdge(MixedVertex):

    def __init__(self, input_size, output_size, vertex_type, reduction, args=None,
                 curr_vtx_id=None, curr_cell_id=None):
        super(DartsSpaceMixedVertexEdge, self).__init__(input_size, output_size, vertex_type,
                                                    args=args, curr_vtx_id=curr_vtx_id)
        self.reduce_ops = None
        self.reduction = reduction
        self.should_reduce = False
        bn_kwargs = {'track_running_stats': args.wsbn_track_stat, 'affine': args.wsbn_affine} if args else {}
        logging.debug("Setting BN-kwargs to {}".format(bn_kwargs))
      
        if reduction:
            # create a separate list of reduced cells.
            ops = {}
            for p in PRIMITIVES:
                op = BN_OPS[p](output_size, 2, bn_kwargs)
                if 'pool' in p:
                    op = nn.Sequential(op, nn.BatchNorm2d(output_size, affine=False))
                ops[p] = op
            self.reduce_ops = nn.ModuleDict(ops)
            self.reduction = True
        ops = {}
        for p in PRIMITIVES:
            op = BN_OPS[p](output_size, 1, bn_kwargs)
            if 'pool' in p:
                op = nn.Sequential(op, nn.BatchNorm2d(output_size, affine=False))
            ops[p] = op
        self.ops = nn.ModuleDict(ops)

    def change_vertex_type(self, vertex_type, proj_op_ids=None):
        """ proj_op_ids is a single element list, that contains the input node id. """
        self.vertex_type = vertex_type
        self.proj_op_ids = proj_op_ids
        # this is input id. 
        if any([i < 2 for i in proj_op_ids]):
            self.should_reduce = True
        else:
            self.should_reduce = False

    @property
    def current_op(self):
        if self.reduction and self.should_reduce:
            return self.reduce_ops[self.vertex_type]
        else:
            return self.ops[self.vertex_type]

    def forward(self, x, weight=None):
        """
        Function it runs is
               relu(bn(w * \sum_i x_i + b))
        :param x:
        :param weight:
        :return:
        """
        # always set this node to Train mode, for batch normalization.
        if weight is not None:
            logging.warning("not yet support reduce_ops, todo later.")
            return sum(w * self.ops[op](x) for w, op in zip(weight, PRIMITIVES))
        else:
            return self.current_op(x)
