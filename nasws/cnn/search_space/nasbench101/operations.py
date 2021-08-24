import logging

import IPython
import torch
try:
    from torch.nn import SyncBatchNorm
except:
    logging.warning("SyncBatchNorm not loaded. Should using pytorch > 1.1.0.")

from itertools import repeat

from functools import partial
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
from torch._six import container_abcs


from nasws.cnn.operations import WSBNFull
from ..api import MixedVertex as MixedVertexTemplate
from .lib import base_ops
from nasws.cnn.operations.ofa.utils import int2list
from nasws.cnn.operations.ofa.ofa_dynamic_ops import KernelTransformDynamicConv2d, DynamicBatchNorm2d

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)

def conv_bn_relu(kernel_size, input_size, output_size):
    padding = 1 if kernel_size == 3 else 0

    out = nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(
            output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON
        ),
        nn.ReLU(inplace=False),
    )

    return out


def maxpool(kernel_size, *spargs):
    return nn.MaxPool2d(kernel_size, stride=1, padding=1)

OPS = {
        "conv1x1-bn-relu": partial(conv_bn_relu, 1),
        "conv3x3-bn-relu": partial(conv_bn_relu, 3),
        "maxpool3x3": partial(maxpool, 3),
    }


# Channel dropout functions.
def channel_select_dropout(x, out_size, dim):
    """ Channel dropout directly, rather than a layer. """
    orig_size = x.size()[dim]
    # move indices to GPU. this will be an interesting point.
    
    indices = torch.randperm(orig_size)[:out_size].to(x.device)
    return torch.index_select(x, dim, indices)


def channel_interpolation(x, out_size, dim):
    """Interpolation channel """
    # orig_size = x.size()[dim]
    # Another idea, is to design a module that based on the input, designing a
    # re-weighting mechanism.
    # Potentially this will increase the capability of representation of network, 
    # but has less degree of freedom than a feature generator.
    # adaptive pooling.
    new_size = list(x.size())
    new_size[dim] = out_size
    return F.adaptive_avg_pool3d(x, new_size[1:])


def channel_random_chunck(x, out_size, dim):
    """ random a chunk of channel size."""
    orig_size = x.size()[dim]
    if orig_size - out_size == 0:
        return x
    start_channel = int(np.random.randint(0, orig_size - out_size))
    out = x.contiguous()
    return out.narrow(dim, start_channel, int(out_size))


def channel_chunck_fix(x, out_size, dim, placeholder=0):
    out = x.contiguous()
    return out.narrow(dim, placeholder, int(out_size))


channel_drop_ops ={
    'fixed_chunck': channel_chunck_fix,
    'fixed': channel_chunck_fix,
    'random_chunck': channel_random_chunck,
    'select_dropout': channel_select_dropout,
    'interpolation': channel_interpolation
}


class ChannelDropout(nn.Module):
    """
    Input:  [:, dim, :, :]
    Output: [:, out_size, :, :]
        out_size < dim, must be true.

    Various methods to test.

    """
    def __init__(self, channel_dropout_method='fixed', channel_dropout_dropouto=0.):
        super(ChannelDropout, self).__init__()
        self.channel_fn = channel_drop_ops[channel_dropout_method]
        self.dropout = nn.Dropout2d(channel_dropout_dropouto) if 0. < channel_dropout_dropouto < 1.0 else None

    def forward(self, input, output_size):
        # passed into a dropout layer and then get the
        # out = super(ChannelDropout, self).forward(input)
        if input.size()[1] - output_size <= 0:
            return input
        # out = input.contiguous()
        # start_channel = int(np.random.randint(0, input.size()[1] - output_size))
        # return out.narrow(1, start_channel, int(output_size))
        out = self.channel_fn(input, output_size, 1)
        if self.dropout:
            out = self.dropout(out)
        return out


class DynamicConv2d(nn.Conv2d):
    """
    Input: [:, dim, :, :], but dim can change.
    So it should dynamically adjust the self.weight.
    Using the similar logic as ChannelDropout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 dynamic_conv_method='fixed', dynamic_conv_dropoutw=0.
                 ):
        self.channel_fn = channel_drop_ops[dynamic_conv_method]
        self.dropout = nn.Dropout2d(dynamic_conv_dropoutw) if 0. < dynamic_conv_dropoutw < 1.0 else None
        
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)

    def forward(self, input):
        batchsize, num_channels, height, width = input.size()
        # print("conv weight shape", self.weight.size())
        # print("conv channel size", self.weight.size()[1])
        # print("target channel ", num_channels)
        w = self.channel_fn(self.weight, num_channels, 1)
        if self.dropout:
            w = self.dropout(w)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            w, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class DynamicConv2dv2(DynamicConv2d):
    """May 8 2020. Update the previous implementation to better align with NasBench original space.

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', dynamic_conv_method='fixed', dynamic_conv_dropoutw=0.0,
    dynamic_in_channel_choices=None, dynamic_out_channel_choices=None):

        if dynamic_conv_method == 'expand':
            assert dynamic_in_channel_choices is not None
            assert dynamic_out_channel_choices is not None
            in_channels = sum(dynamic_in_channel_choices)
            out_channels = sum(dynamic_out_channel_choices)
            def expand_channel_fn(weight, target_size, dimension):
                '''This will raise IndexError 
                    if the target size is not in dynamic channel choises
                '''
                sort_dims = dynamic_in_channel_choices if dimension == 1 else dynamic_out_channel_choices
                p_ind = sort_dims.index(target_size)
                placeholder = sum(sort_dims[:p_ind])
                return channel_chunck_fix(weight, target_size, dimension, placeholder)
            # add this to the chanel drop ops dict
            channel_drop_ops['expand'] = expand_channel_fn
            logging.debug(f'Creating expand channels: the size is {in_channels, out_channels}')
            
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, dynamic_conv_method=dynamic_conv_method, dynamic_conv_dropoutw=dynamic_conv_dropoutw)
        self.in_size = in_channels
        self.out_size = out_channels
        
    def forward(self, input):
        batchsize, num_channels, height, width = input.size()
        w = self.channel_fn(self.weight, self.out_size, 0)
        w = self.channel_fn(w, self.in_size, 1)

        if self.dropout:
            w = self.dropout(w)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            w, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def change_size(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        

class DynamicReLUConvBN(nn.Module):
    """
    Conv Bn relu for weight sharing.
        Because the intrinsic of NasBench, for each cell,
            output_size = sum(intermediate_node.output_size)

        So to make weight sharing possible, we need to create a operation with full channel size then prune the channel.
        Each time, about the forward operation, we need to specify the output-size manually, for different architecture.

    """
    def __init__(self, kernel_size, full_input_size, full_output_size, curr_vtx_id=None, args=None):
        super(DynamicReLUConvBN, self).__init__()
        self.args = args
        padding = 1 if kernel_size == 3 else 0
        # assign layers.
        self.relu = nn.ReLU(inplace=False)
        self.conv = DynamicConv2d(
            full_input_size, full_output_size, kernel_size, padding=padding, bias=False,
            dynamic_conv_method=args.dynamic_conv_method, dynamic_conv_dropoutw=args.dynamic_conv_dropoutw
        )
        self.curr_vtx_id = curr_vtx_id
        
        if args.wsbn_sync:
            # logging.debug("Using sync bn.")
            self.bn = SyncBatchNorm(full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON,
                                    track_running_stats=args.wsbn_track_stat,
                                    affine=args.wsbn_affine)
        else:
            self.bn = nn.BatchNorm2d(full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON,
                                     track_running_stats=args.wsbn_track_stat,
                                     affine=args.wsbn_affine)
        # for dynamic channel
        self.channel_drop = ChannelDropout(args.channel_dropout_method, args.channel_dropout_dropouto)
        self.output_size = full_output_size
        self.current_outsize = full_output_size # may change according to different value.
        self.current_insize = full_input_size

    def forward(self, x, bn_train=False, output_size=None):
        # compute and only output the
        output_size = output_size or self.current_outsize
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.channel_drop(x, output_size)
        return x

    def change_size(self, in_size, out_size):
        self.current_insize = in_size
        self.current_outsize = out_size


class DynamicReLUConvBNv2(nn.Module):
    def __init__(self, kernel_size, full_input_size, full_output_size, curr_vtx_id=None, args=None, **kwargs):
        super(DynamicReLUConvBNv2, self).__init__()
        self.args = args
        padding = 1 if kernel_size == 3 else 0
        # assign layers.
        self.curr_vtx_id = curr_vtx_id        
        
        # for dynamic channel
        self.relu = nn.ReLU(inplace=False)
        self.output_size = full_output_size
        self.current_outsize = full_output_size # may change according to different value.
        self.current_insize = full_input_size
        self.conv = DynamicConv2dv2(
            full_input_size, full_output_size, kernel_size, padding=padding, bias=False,
            dynamic_conv_method=args.dynamic_conv_method, dynamic_conv_dropoutw=args.dynamic_conv_dropoutw,
            **kwargs
        )
        
        if args.wsbn_sync:
            # logging.debug("Using sync bn.")
            raise NotImplementedError("Currently not implemented for v2")
        else:
            self.bn = DynamicBatchNorm2d(full_output_size, 
                momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON,
                track_running_stats=args.wsbn_track_stat,
                affine=args.wsbn_affine
                )

    def forward(self, x, bn_train=False, output_size=None):
        # compute and only output the
        input_size = x.size()[1]
        output_size = output_size or self.current_outsize
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)        
        assert input_size == self.current_insize, ValueError(f'Size mismatch for DynamicLayer {input_size} != {self.current_insize}')
        assert x.size()[1] == output_size, ValueError(f'Size mismatch for DynamicLayer {x.size()[1]} != {output_size}')
        return x

    def change_size(self, in_size:int, out_size:int):
        self.current_insize = in_size
        self.current_outsize = out_size
        self.conv.change_size(in_size, out_size)


class DynamicConvWSBNRelu(DynamicReLUConvBN):

    def __init__(self, kernel_size, full_input_size, full_output_size, curr_vtx_id=None, args=None):
        super(DynamicConvWSBNRelu, self).__init__(kernel_size, full_input_size, full_output_size, curr_vtx_id, args)
        del self.bn
        assert curr_vtx_id is not None and 0 <= curr_vtx_id < args.num_intermediate_nodes + 2, \
            logging.error("getting vertex id {}".format(curr_vtx_id))
        
        self.bn = WSBNFull(2 ** curr_vtx_id - 1, full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON)
        logging.debug('construction WSBN before creation, current id {}, possible input config {}'.format(curr_vtx_id, 2 ** curr_vtx_id - 1))
        self.previous_node_id = 0
        self.num_possible_inputs = 2 ** curr_vtx_id - 1

    def change_previous_vertex_id(self, _id):
        """ change the id to use WSBN """
        if 0 <= _id < self.num_possible_inputs:
            self.previous_node_id = _id
        else:
            # IPython.embed()
            raise ValueError(f"Assigning previous id {_id} wrongly, accept {self.num_possible_inputs}")

    def forward(self, x, bn_train=False, output_size=None):
        output_size = output_size or self.current_outsize
        x = self.conv(x)
        if self.bn_train and self.training:
            self.bn.train()
        x = self.bn(x, self.previous_node_id) # always using True for the moment.
        return x


class DynamicConvDifferentNorm(DynamicReLUConvBN):
    def __init__(self, kernel_size, full_input_size, full_output_size, curr_vtx_id=None, args=None, norm_type='instance'):
        super(DynamicConvDifferentNorm, self).__init__(kernel_size, full_input_size, full_output_size, curr_vtx_id, args)
        del self.bn
        if norm_type == 'instance':
            self.bn = nn.InstanceNorm2d(full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON)
        # elif norm_type == 'layer': # need to calculate the size, so abandon here.
        #     self.bn = nn.LayerNorm()
        elif norm_type == 'group':
            self.bn = nn.GroupNorm(num_groups=8, num_channels=full_output_size, eps=base_ops.BN_EPSILON)
        elif norm_type == 'local':
            self.bn = nn.LocalResponseNorm(2)
        else:
            raise ValueError("Norm type not yet supported ", norm_type)
        # self.bn = WSBNFull(curr_vtx_id + 1, full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON)
        logging.debug('construction {} before creation, current id {}'.format(norm_type, curr_vtx_id))
        self.previous_node_id = 0
        self.num_possible_inputs = curr_vtx_id + 1



class DynamicReLUConvBNOFA(nn.Module):
    """ OFA NASBench-101 implementation here. """
    def __init__(self, kernel_size, full_input_size, full_output_size, curr_vtx_id=None, args=None):
        """
        Kernel size list here indicating the max kernel size. it should be generating the kernel_size list accordingly.
        """
        super(DynamicReLUConvBNOFA, self).__init__()
        self.args = args
        self.kernel_size_list = list(range(1, kernel_size + 1, 2))
        self.relu = nn.ReLU(inplace=False)
        self.conv = KernelTransformDynamicConv2d(
            full_input_size, self.kernel_size_list, stride=1, dilation=1, 
            transfer_mode=args.ofa_parameter_transform)
        self.curr_vtx_id = curr_vtx_id
        tracking_stat = args.wsbn_track_stat
        if args.wsbn_sync:
            # logging.debug("Using sync bn.")
            raise NotImplementedError("Do not support wsbn sync yet.")
        else:
            self.bn = DynamicBatchNorm2d(full_output_size)
            # self.bn = nn.BatchNorm2d(full_output_size, momentum=base_ops.BN_MOMENTUM, eps=base_ops.BN_EPSILON,
                                    #  track_running_stats=tracking_stat)

        self.bn_train = args.wsbn_train     # store the bn train of not.
        if self.bn_train:
            self.bn.train()
        else:
            self.bn.eval()

        # for dynamic channel
        # self.channel_drop = ChannelDropout(args.channel_dropout_method, args.channel_dropout_dropouto)
        self.output_size = full_output_size
        self.current_outsize = full_output_size # may change according to different value.
        self.current_insize = full_input_size
    
    def forward(self, x, bn_train=False, output_size=None):
        # compute and only output the
        output_size = output_size or self.current_outsize
        x = self.conv(x)
        # if self.bn_train and self.training:
        self.bn.train()
        x = self.bn(x)
        x = self.relu(x)
        # x = self.channel_drop(x, output_size)
        return x

    def change_size(self, in_size, out_size):
        self.current_insize = in_size
        self.current_outsize = out_size

    def change_kernel_size(self, kernel_size):
        self.conv.active_kernel_size = kernel_size

def maxpool_search(kernel_size, input_size, output_size, curr_vtx_id, args):
    return nn.MaxPool2d(kernel_size, stride=1, padding=1)


SEARCH_OPS = {
        "conv1x1-bn-relu": partial(DynamicReLUConvBN, 1),
        "conv3x3-bn-relu": partial(DynamicReLUConvBN, 3),
        "maxpool3x3": partial(maxpool_search, 3),
    }

SEARCH_OPS_v2 = {
        "conv1x1-bn-relu": partial(DynamicReLUConvBNv2, 1),
        "conv3x3-bn-relu": partial(DynamicReLUConvBNv2, 3),
        "maxpool3x3": partial(maxpool_search, 3),
    }


class Truncate(nn.Module):
    def __init__(self, channels):
        super(Truncate, self).__init__()
        self.channels = channels

    def forward(self, x):
        return x[:, : self.channels]


class MixedVertex(MixedVertexTemplate):

    oper = SEARCH_OPS_v2

    def __init__(self, input_size, output_size, vertex_type, do_projection, args=None,
                 curr_vtx_id=None, curr_cell_id=None, 
                 dynamic_in_channel_choices=None, dynamic_out_channel_choices=None):
        """

        :param input_size: input size to the cell, == in_channels for input node, and output_channels otherwise.
        :param output_size: full output size, should be the same as cell output_channels
        :param vertex_type: initial vertex type, choices over Ops.keys.
        :param do_projection: Projection with parametric operation. Set True for input and false for others
        :param args: args parsed into
        :param curr_vtx_id: current vertex id. used for advance logic for each cell.
        :param dynamic_in_channel_choices(optional): used for expand layer
        :param dynamic_out_channel_choices(optional): used for expand layer
        """
        assert vertex_type in SEARCH_OPS_v2.keys(), 'vertex must be inside OPS keys'
        super(MixedVertex, self).__init__(input_size, output_size, vertex_type,
                                          do_projection, args, curr_vtx_id, curr_cell_id)

        # list to store the current projection operations, may change with model_spec
        self.proj_ops = nn.ModuleList()
        self._current_proj_ops = []
        for ind, (in_size, do_proj) in enumerate(zip(input_size, do_projection)):
            proj_op = self.oper["conv1x1-bn-relu"](in_size, output_size, curr_vtx_id=curr_vtx_id, args=args, dynamic_in_channel_choices=[in_size], dynamic_out_channel_choices=dynamic_out_channel_choices) \
                if do_proj else Truncate(output_size)
            self.proj_ops.append(proj_op)
            self._current_proj_ops.append(ind)
        # create the ops.
        ops = nn.ModuleDict([])
        for k in self.oper.keys():
            if 'conv' in k:
                ops[k] = self.oper[k](output_size, output_size, curr_vtx_id=curr_vtx_id, args=args,
                    dynamic_in_channel_choices=dynamic_in_channel_choices, dynamic_out_channel_choices=dynamic_out_channel_choices)
            else:
                ops[k] = self.oper[k](output_size, output_size, curr_vtx_id=curr_vtx_id, args=args)
        self.ops = ops

    def change_vertex_type(self, input_size, output_size, vertex_type, proj_op_ids=None):
        super(MixedVertex, self).change_vertex_type(input_size, output_size, vertex_type, proj_op_ids)
        # for nasbench only, change the convolutional channels accordingly.
        for k, op in self.ops.items():
            # for convolutional layers, change the input/output size accordingly.
            if 'conv' in k:
                op.change_size(output_size, output_size)

    def forward(self, x, weight=None):
        """
        Forward function with weighted sum support.
               
        Intrinsically, this function computes based on the active 
            
            inputs: previous node outputs [in, n1, n2, ...]

            y = sum([project_op(i) for i in inputs])
            output = current_op(y)
            

        :param x: previous node outputs [in, n1, n2, ...]
        :param weight: TODO
        :return: tensor, see above
        """
        if weight is not None and self.soft_aggregation:
            raise RuntimeError("Please call self.forward_darts for DARTS weighted operation.")
            # pass
        else:
            proj_iter = iter(self.current_proj_ops)
            input_iter = iter(x)
            try:
                out = next(proj_iter)(next(input_iter))
                for proj, inp in zip(proj_iter, input_iter):
                    out = out + proj(inp)
            except RuntimeError as e:
                IPython.embed()
            return self.current_op(out)

    def forward_darts(self, x, weight):
        """

        :param x: input.
        :param weight: topology, op weights accordingly
        :return:
        """
        topo_weight, op_weight = weight
        # This is the Single-path way.
        # apply all projection to get real input to this vertex
        proj_iter = iter(self.current_proj_ops)
        input_iter = iter(x)
        out = next(proj_iter)(next(input_iter))
        for ind, (proj, inp) in enumerate(zip(proj_iter, input_iter)):
            out = out + topo_weight[ind] * proj(inp)
        # no such thing called current op.
        output = 0.0
        try:
            for ind, op in enumerate(self.ops.values()):
                output = output + op_weight[ind] * op(out)
        except RuntimeError as e:
            IPython.embed()
        return output

class MixedVertexEdge(MixedVertex):

    def __init__(self, input_size, output_size, vertex_type, do_projection, args=None,
                 curr_vtx_id=None, curr_cell_id=None):
        """

        :param input_size: input size to the cell, == in_channels for input node, and output_channels otherwise.
        :param output_size: full output size, should be the same as cell output_channels
        :param vertex_type: initial vertex type, choices over Ops.keys.
        :param do_projection: Projection with parametric operation. Set True for input and false for others
        :param args: args parsed into
        :param curr_vtx_id: current vertex id. used for advance logic for each cell.
        """
        assert vertex_type in SEARCH_OPS.keys(), 'vertex must be inside OPS keys'
        super(MixedVertex, self).__init__(input_size, output_size, vertex_type,
                                          do_projection, args, curr_vtx_id, curr_cell_id)

        # list to store the current projection operations, may change with model_spec
        self.proj_ops = nn.ModuleList()
        self._current_proj_ops = []
        for ind, (in_size, do_proj) in enumerate(zip(input_size, do_projection)):
            proj_op = self.oper["conv1x1-bn-relu"](in_size, output_size, curr_vtx_id=curr_vtx_id, args=args) \
                if do_proj else Truncate(output_size)
            self.proj_ops.append(proj_op)
            self._current_proj_ops.append(ind)
        # create the ops.
        # curr_vtx_id must be valid
        self.ops = nn.ModuleDict()
        for ind in range(curr_vtx_id):
            edge_k = "{:}<-{:}".format(ind, curr_vtx_id)
            _ops = nn.ModuleDict()
            for k in SEARCH_OPS.keys():
                _ops[k] = self.oper[k](output_size, output_size, curr_vtx_id=curr_vtx_id, args=args)
            self.ops[edge_k] = _ops

    @property
    def current_op(self):
        return self.ops[self.vertex_type]

class MixedVertexOFA(MixedVertex):
    """ Implement the OFA version of MixedVertex for NASBench 101. 
        This does not generalize. make another version later.

        OFA here indicates "Once for all" paper. It uses a dynamic convolutional 
        channels to merge the active weights by projecting a matrix.
        This needs more work to verify the correctness. and align with the model_specs.

    """
    oper = {
        "projection": partial(DynamicReLUConvBN, 1),
        "conv-bn-relu": partial(DynamicReLUConvBNOFA, 3),
        "maxpool3x3": partial(maxpool_search, 3),
    }

    original_keys = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    kernel_size_map = {'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 3}

    def __init__(self, input_size, output_size, vertex_type, do_projection, args=None,
                curr_vtx_id=None, curr_cell_id=None):
        """
        wiring 3x3 and 1x1 to the same op. change the internel logic only..

        :param input_size: input size to the cell, == in_channels for input node, and output_channels otherwise.
        :param output_size: full output size, should be the same as cell output_channels
        :param vertex_type: initial vertex type, choices over Ops.keys.
        :param do_projection: Projection with parametric operation. Set True for input and false for others
        :param args: args parsed into
        :param curr_vtx_id: current vertex id. used for advance logic for each cell.
        """

        assert vertex_type in self.original_keys, 'vertex must be inside OPS keys'
        super(MixedVertex, self).__init__(input_size, output_size, vertex_type,
                                          do_projection, args, curr_vtx_id, curr_cell_id)
        
        # Override the max kernel size accordingly.
        self.oper['conv-bn-relu'] = partial(DynamicReLUConvBNOFA, args.ofa_max_kernel_size)
        # list to store the current projection operations, may change with model_spec
        self.proj_ops = nn.ModuleList()
        self._current_proj_ops = []
        for ind, (in_size, do_proj) in enumerate(zip(input_size, do_projection)):
            proj_op = self.oper["projection"](in_size, output_size, curr_vtx_id=curr_vtx_id, args=args) \
                if do_proj else Truncate(output_size)
            self.proj_ops.append(proj_op)
            self._current_proj_ops.append(ind)

        # create the ops.
        ops = nn.ModuleDict([])
        for k in ['conv-bn-relu', 'maxpool3x3']:
            ops[k] = self.oper[k](output_size, output_size, curr_vtx_id=curr_vtx_id, args=args)
        self.ops = ops

    @property
    def current_op(self):
        """ modify the current op to support dynamic conv projection """
        if 'conv' in self.vertex_type:
            kernel_proj_conv = self.ops['conv-bn-relu']
            kernel_proj_conv.change_kernel_size(self.kernel_size_map[self.vertex_type])
            return kernel_proj_conv
        else:
            return self.ops[self.vertex_type]

    def change_vertex_type(self, input_size, output_size, vertex_type, proj_op_ids=None):
        """ wrap the vertex type according to internel """
        _vertex_type = 'conv-bn-relu' if 'conv' in vertex_type else vertex_type
        super().change_vertex_type(input_size, output_size, _vertex_type, proj_op_ids=proj_op_ids)
        self.vertex_type = vertex_type


class MixedVertexNAO(MixedVertex):

    def __init__(self, input_size, output_size, vertex_type, do_projection, args=None,
                 curr_vtx_id=None, curr_cell_id=None):
        super(MixedVertexNAO, self).__init__(input_size, output_size, vertex_type, do_projection, args,
                 curr_vtx_id, curr_cell_id)
        self.drop_path_keep_prob = self.args.child_drop_path_keep_prob
        self.cells = self.args.layers
        self.nodes = self.args.num_intermediate_nodes

    def forward(self, x, weight=None):
        if weight is not None and self.soft_aggregation:
            raise NotImplementedError("To support DARTS way later.")
            # pass
        else:
            # This is the Single-path way.
            # apply all projection to get real input to this vertex
            proj_iter = iter(self.current_proj_ops)
            input_iter = iter(x)
            out = next(proj_iter)(next(input_iter))
            for proj, inp in zip(proj_iter, input_iter):
                out = out + proj(inp)
            output = self.current_op(out)
            if not 'pool' in self.vertex_type and self.training:
                output = self.apply_drop_path(output,
                                              self.drop_path_keep_prob, self.curr_cell_id, self.cells,
                                              self.curr_vtx_id,
                                              self.nodes)
            return output
    
    @staticmethod
    def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
        layer_ratio = float(layer_id+1) / (layers)
        drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
        step_ratio = float(step + 1) / steps
        drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
        if drop_path_keep_prob < 1.:
            mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
            # x.div_(drop_path_keep_prob)
            # x.mul_(mask)
            x = x / drop_path_keep_prob * mask
        return x


class MixedVertexDARTS(MixedVertex):

    def forward(self, x, weight):
        """

        :param x: input.
        :param weight: topology, op weights accordingly
        :return:
        """
        topo_weight, op_weight = weight
        # This is the Single-path way.
        # apply all projection to get real input to this vertex
        proj_iter = iter(self.current_proj_ops)
        input_iter = iter(x)
        out = next(proj_iter)(next(input_iter))
        for ind, (proj, inp) in enumerate(zip(proj_iter, input_iter)):
            out = out + topo_weight[ind] * proj(inp)
        # no such thing called current op.
        output = 0.0
        try:
            for ind, op in enumerate(self.ops.values()):
                output = output + op_weight[ind] * op(out)
        except RuntimeError as e:
            IPython.embed()
        return output


class MixedVertexWSBN(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvWSBNRelu, 1),
        "conv3x3-bn-relu": partial(DynamicConvWSBNRelu, 3),
        "maxpool3x3": partial(maxpool_search, 3),
    }

    def forward(self, x, weight=None):
        if weight is not None and self.soft_aggregation:
            raise NotImplementedError("To support DARTS way later.")
        else:
            # version 1. this is like
            #       relu( \sum_i{bn_i ( w x_i + b_i))} )

            # it is wrong, current_op change position by doing so.
            # out = 0.
            # for in_id, inp in zip(self._current_proj_ops, x):
            #     if 'conv' in self.vertex_type:
            #         self.current_op.change_previous_vertex_id(in_id)
            #     out = out + self.current_op(self.proj_ops[in_id](inp))
            # return F.relu(out)

            # Correct version
            proj_iter = iter(self.current_proj_ops)
            input_iter = iter(x)
            out = next(proj_iter)(next(input_iter))
            for proj, inp in zip(proj_iter, input_iter):
                out = out + proj(inp)
            if 'conv'  in self.vertex_type:
                prev_wsbn_config = sum([2 ** int(i) for i in self._current_proj_ops]) - 1
                self.current_op.change_previous_vertex_id(prev_wsbn_config)
            return self.current_op(out)


class MixedVertexWSBNSumEnd(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvWSBNRelu, 1),
        "conv3x3-bn-relu": partial(DynamicConvWSBNRelu, 3),
        "maxpool3x3": partial(maxpool_search, 3),
    }

    def forward(self, x, weight=None):
        if weight is not None and self.soft_aggregation:
            raise NotImplementedError("To support DARTS way later.")
        else:
            # version 2. this is like
            #        \sum_i relu({bn_i ( w x_i + b_i))})
            out = 0.
            for in_id, inp in zip(self._current_proj_ops, x):
                if 'conv' in self.vertex_type:
                    self.current_op.change_previous_vertex_id(in_id)
                out = out + F.relu(self.current_op(self.proj_ops[in_id](inp)))
            return out


class MixedVertexInstanceNorm(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvDifferentNorm, 1, norm_type='instance'),
        "conv3x3-bn-relu": partial(DynamicConvDifferentNorm, 3, norm_type='instance'),
        "maxpool3x3": partial(maxpool_search, 3),
    }


class MixedVertexGroupNorm(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvDifferentNorm, 1, norm_type='group'),
        "conv3x3-bn-relu": partial(DynamicConvDifferentNorm, 3, norm_type='group'),
        "maxpool3x3": partial(maxpool_search, 3),
    }


class MixedVertexLRNorm(MixedVertex):
    oper = {
        "conv1x1-bn-relu": partial(DynamicConvDifferentNorm, 1, norm_type='local'),
        "conv3x3-bn-relu": partial(DynamicConvDifferentNorm, 3, norm_type='local'),
        "maxpool3x3": partial(maxpool_search, 3),
    }


nasbench_vertex_weight_sharing = {
    'mixedvertex':     MixedVertex,
    'mixedvertex_nao':     MixedVertexNAO,  # add this apply drop path.
    'mixedvertex_instance_norm':     MixedVertexInstanceNorm,
    'mixedvertex_group_norm':     MixedVertexGroupNorm,
    'mixedvertex_local_norm':     MixedVertexLRNorm,
    'mixedvertex_wsbn':     MixedVertexWSBN,
    'mixedvertex_wsbn_endsum':     MixedVertexWSBNSumEnd,
    'nao_ws':  MixedVertex,
    'enas_ws': MixedVertex,
    'darts_ws': MixedVertexDARTS,
    'mixedvertex_ofa': MixedVertexOFA,
}

nasbench_edge_weight_sharing = {
    'mixedvertex': MixedVertex,
}