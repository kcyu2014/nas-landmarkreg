import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch

from .utils.network_utils import get_same_padding, sub_filter_start_end
from .utils.pytorch_modules import make_divisible, SEModule
import logging
import IPython


class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = None  # None or 1
    
    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()
        
        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        self.padding = None

        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.max_in_channels, bias=False,
        )
        
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)
    
    def get_active_filter(self, in_channel, kernel_size, out_channel=None):
        """ Filter transformation, make the input channel size. """
        out_channel = out_channel if out_channel else in_channel 
        max_kernel_size = max(self.kernel_size_list)
        
        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        # logging.debug(f"kernel size set {self._ks_set}")
        # logging.debug(f'conv kernel start {start}, end {end}, m {max_kernel_size} k {kernel_size}')
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        # logging.debug(f'conv kernel size {filters.size()}')
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                # input_filter size (target_kernel_size ** 2, in_channel * out_channels)
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                # crucial difference, adding a parameter adaptor.
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters
    
    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        
        # add an option to override padding.
        padding = self.padding or get_same_padding(kernel_size)
        
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, in_channel
        )
        # print(kernel_size, padding, self.stride, self.dilation, x.size(), y.size())
        return y


class KernelTransformDynamicConv2d(DynamicSeparableConv2d):
    
    KERNEL_TRANSFORM_MODE = 1
    groups = 1

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1, transfer_mode=1):
        super(DynamicSeparableConv2d, self).__init__()
        self.KERNEL_TRANSFORM_MODE = transfer_mode if transfer_mode == 1 else None
        self.max_in_channels = max_in_channels
        # self.max_out_channels = max_out_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_in_channels, max(self.kernel_size_list), self.stride,
            groups=self.groups, bias=False,
        )
        
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix, 3to1_matrix or so.
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        
        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            x, filters, None, self.stride, padding, self.dilation, self.groups
        )
        return y



class DynamicPointConv2d(nn.Module):
    """ Same as SPOS one, using fixed channel size. """
    def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
        super(DynamicPointConv2d, self).__init__()
        
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        self.conv = nn.Conv2d(
            self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
        )
        
        self.active_out_channel = self.max_out_channels
    
    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()
        
        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicLinear(nn.Module):
    """ again, this is a similar dynamic linear with fixed first chunck. """
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()
        
        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias
        
        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)
        
        self.active_out_features = self.max_out_features
    
    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features
        
        in_features = x.size(1)
        weight = self.linear.weight[:out_features, :in_features].contiguous()
        bias = self.linear.bias[:out_features] if self.bias else None
        y = F.linear(x, weight, bias)
        return y


class DynamicBatchNorm2d(nn.Module):
    """ This is interesting. Another version of my current implementation. """
    SET_RUNNING_STATISTICS = False
    
    def __init__(self, max_feature_dim, **bn_kwargs):
        super(DynamicBatchNorm2d, self).__init__()
        # logging.debug(f'DynamicBN kwargs {bn_kwargs}')
        self.max_feature_dim = max_feature_dim 
        self.bn = nn.BatchNorm2d(self.max_feature_dim, **bn_kwargs)
    
    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):

        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            # this is funny. so let's consider to redo 

            running_mean = bn.running_mean[:feature_dim] if bn.running_mean is not None else None
            running_var = bn.running_var[:feature_dim] if bn.running_var is not None else None
            weight = bn.weight[:feature_dim] if bn.weight is not None else None
            bias = bn.bias[:feature_dim] if bn.bias is not None else None

            exponential_average_factor = 0.0
            
            if bn.training and bn.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            
            return F.batch_norm(
                x, running_mean, running_var, weight, bias, 
                bn.training or not bn.track_running_stats,
                exponential_average_factor, bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicSE(SEModule):
    
    def __init__(self, max_channel):
        super(DynamicSE, self).__init__(max_channel)

    def forward(self, x):
        in_channel = x.size(1)
        num_mid = make_divisible(in_channel // self.reduction, divisor=8)

        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        # reduce
        reduce_conv = self.fc.reduce
        reduce_filter = reduce_conv.weight[:num_mid, :in_channel, :, :].contiguous()
        reduce_bias = reduce_conv.bias[:num_mid] if reduce_conv.bias is not None else None
        y = F.conv2d(y, reduce_filter, reduce_bias, 1, 0, 1, 1)
        # relu
        y = self.fc.relu(y)
        # expand
        expand_conv = self.fc.expand
        expand_filter = expand_conv.weight[:in_channel, :num_mid, :, :].contiguous()
        expand_bias = expand_conv.bias[:in_channel] if expand_conv.bias is not None else None
        y = F.conv2d(y, expand_filter, expand_bias, 1, 0, 1, 1)
        # hard sigmoid
        y = self.fc.h_sigmoid(y)

        return x * y
