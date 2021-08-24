import logging

# define some operations
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

__all__ = ['OPS', 'ReLUConvBN', 'ResNetBasicblock',
           'SearchSpaceNames', 'WSBN_OPS']

OPS = {
    'none': lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride: POOLING(C_in, C_out, stride, 'avg'),
    'max_pool_3x3': lambda C_in, C_out, stride: POOLING(C_in, C_out, stride, 'max'),
    'nor_conv_7x7': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (7, 7), (stride, stride), (3, 3), (1, 1)),
    'nor_conv_3x3': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1)),
    'nor_conv_1x1': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1)),
    'skip_connect': lambda C_in, C_out, stride: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride),
}

UPSAMPLE = {
    'bilinear': lambda C_in, C_out, stride: nn.Upsample(scale_factor=stride, mode='bilinear'),
    'bicubic': lambda C_in, C_out, stride: nn.Upsample(scale_factor=stride, mode='bicubic'),
    'convtranspose': lambda C_in, C_out, stride: nn.ConvTranspose2d(C_in, C_out, kernel_size=4, stride=stride, padding=1),
    'pixelshuffle': lambda C_in, C_out, stride: PixelShuffleLayer(stride, C_in, C_out)
}


WSBN_OPS = {
    'none': lambda C_in, C_out, stride, bn_kwargs: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, bn_kwargs: POOLING(C_in, C_out, stride, 'avg'),
    'max_pool_3x3': lambda C_in, C_out, stride, bn_kwargs: POOLING(C_in, C_out, stride, 'max'),
    'nor_conv_7x7': lambda C_in, C_out, stride, bn_kwargs: ReLUConvBN(C_in, C_out, (7, 7), (stride, stride), (3, 3), (1, 1), bn_kwargs=bn_kwargs),
    'nor_conv_3x3': lambda C_in, C_out, stride, bn_kwargs: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1), bn_kwargs=bn_kwargs),
    'nor_conv_1x1': lambda C_in, C_out, stride, bn_kwargs: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1), bn_kwargs=bn_kwargs),
    'skip_connect': lambda C_in, C_out, stride, bn_kwargs: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, bn_kwargs),
}


# WSBNOPS = {
#   'none'         : lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
#   'avg_pool_3x3' : lambda C_in, C_out, stride: POOLING(C_in, C_out, stride, 'avg'),
#   'max_pool_3x3' : lambda C_in, C_out, stride: POOLING(C_in, C_out, stride, 'max'),
#   'nor_conv_7x7' : partial(ReLUConvWSBN, kernel_size=(7,7), padding=(3,3), dialation=(1,1)),
#   # 'nor_conv_7x7' : lambda C_in, C_out, stride: ReLUConvWSBN(C_in, C_out, (7,7), (stride,stride), (3,3), (1,1)),
#   'nor_conv_3x3' : partial(ReLUConvWSBN, kernel_size=(3,3), padding=(1,1), dialation=(1,1)),
#   # 'nor_conv_3x3' : lambda C_in, C_out, stride: ReLUConvWSBN(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1)),
#   'nor_conv_1x1' : partial(ReLUConvWSBN, kernel_size=(1,1), padding=(0,0), dialation=(1,1)),
#   # 'nor_conv_1x1' : lambda C_in, C_out, stride: ReLUConvWSBN(C_in, C_out, (1,1), (stride,stride), (0,0), (1,1)),
#   'skip_connect' : lambda C_in, C_out, stride, num_possible_input: Identity() if stride == 1 and C_in == C_out else FactorizedReduceWSBN(C_in, C_out, stride, num_possible_input),
# }

CONNECT_NAS_BENCHMARK = ['none', 'skip_connect', 'nor_conv_3x3']
AA_NAS_BENCHMARK = ['none', 'skip_connect',
                    'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

SearchSpaceNames = {'connect-nas': CONNECT_NAS_BENCHMARK,
                    'aa-nas': AA_NAS_BENCHMARK,
                    'full': sorted(list(OPS.keys()))}


class PixelShuffleLayer(nn.Module):
    def __init__(self, upscale_factor, C_in, C_out):
        super(PixelShuffleLayer, self).__init__()
        self.conv = ReLUConvBN(C_in, C_out * upscale_factor**2, (1, 1), (1, 1), (0, 0), (1, 1))
        self.shift = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x):
        output = self.conv(x)
        return self.shift(output)


class Interpolate(nn.Module):
    def __init__(self, mode, scale_factor=2):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x,  scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x
 
class WSBNFull(nn.Module):

    def __init__(self, num_possible_inputs, num_features, eps=1e-5, momentum=0.1, affine=True):
        """
        Based on WSBN from NAO, but extend to running mean and var.

        :param num_possible_inputs:
        :param num_features:
        :param eps:
        :param momentum:
        :param affine:
        """
        super(WSBNFull, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.ParameterList(
                [nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
            self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for idx in range(num_possible_inputs):
            self.register_buffer('running_mean_{}'.format(idx), torch.zeros(num_features))
            self.register_buffer('running_var_{}'.format(idx), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_possible_inputs):
            getattr(self, f'running_mean_{i}').fill_(1)
            # print(f'running_mean_{i} init ', getattr(self, f'running_mean_{i}').mean())
            getattr(self, f'running_var_{i}').zero_()
            # print(f'running_var_{i} init ', getattr(self, f'running_var_{i}').mean())
            if self.affine:
                self.weight[i].data.fill_(1)
                self.bias[i].data.zero_()

    def forward(self, x, x_id, bn_train=False):
        training = self.training
        if bn_train:
            training = True
        # diff = getattr(self, f'running_mean_{x_id}').sum().item()
        if self.affine:
            return F.batch_norm(
                x, getattr(self, f'running_mean_{x_id}'), getattr(self, f'running_var_{x_id}'),
                self.weight[x_id], self.bias[x_id],
                training, self.momentum, self.eps)
        else:
            return F.batch_norm(
                x, getattr(self, f'running_mean_{x_id}'), getattr(self, f'running_var_{x_id}'),
                None, None,
                training, self.momentum, self.eps)
        # diff -= getattr(self, f'running_mean_{x_id}').sum().item()
        # return o

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__))


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, bn_kwargs={}):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, **bn_kwargs)
        )

    def forward(self, x):
        # Modified for always train mode for NASBench102
        self.train()
        return self.op(x)


class ReLUConvWSBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, num_possible_input):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
        )
        self.bn = WSBNFull(num_possible_input, C_out)

    def forward(self, x, prev_id):
        # Modified for always train mode for NASBench102
        # self.train()
        out = self.op(x)
        out = self.bn(x, prev_id)
        return out


class ResNetBasicblock(nn.Module):

    def __init__(self, inplanes, planes, stride, args=None):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.args = args
        bn_kwargs = {'track_running_stats': args.wsbn_track_stat, 'affine': args.wsbn_affine} if args else {}
        logging.debug("Resblock Setting BN-kwargs to {}".format(bn_kwargs))
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, bn_kwargs)
        self.conv_b = ReLUConvBN(planes, planes, 3,      1, 1, 1, bn_kwargs)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, bn_kwargs)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(
            name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):

    def __init__(self, C_in, C_out, stride, mode):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0)
        if mode == 'avg':
            self.op = nn.AvgPool2d(
                3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max':
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError('Invalid mode={:} in POOLING'.format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, bn_kwargs={}):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            #assert C_out % 2 == 0, 'C_out : {:}'.format(C_out)
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))

        self.bn = nn.BatchNorm2d(C_out, **bn_kwargs)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat(
            [self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduceWSBN(FactorizedReduce):

    def __init__(self, C_in, C_out, stride, num_possible_input):
        super(FactorizedReduce, self).__init__(C_in, C_out, stride)
        self.bn = WSBNFull(num_possible_input, C_out)

    def forward(self, x, prev_id):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat(
            [self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out, prev_id)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


# Auto-ReID: Searching for a Part-Aware ConvNet for Person Re-Identification, ICCV 2019
class PartAwareOp(nn.Module):

    def __init__(self, C_in, C_out, stride, part=4):
        super().__init__()
        self.part = 4
        self.hidden = C_in // 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv_list = nn.ModuleList()
        for i in range(self.part):
            self.local_conv_list.append(
                nn.Sequential(nn.ReLU(), nn.Conv2d(
                    C_in, self.hidden, 1), nn.BatchNorm2d(self.hidden, affine=True))
            )
        self.W_K = nn.Linear(self.hidden, self.hidden)
        self.W_Q = nn.Linear(self.hidden, self.hidden)

        if stride == 2:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 2)
        elif stride == 1:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 1)
        else:
            raise ValueError('Invalid Stride : {:}'.format(stride))

    def forward(self, x):
        batch, C, H, W = x.size()
        assert H >= self.part, 'input size too small : {:} vs {:}'.format(
            x.shape, self.part)
        IHs = [0]
        for i in range(self.part):
            IHs.append(min(H, int((i+1)*(float(H)/self.part))))
        local_feat_list = []
        for i in range(self.part):
            feature = x[:, :, IHs[i]:IHs[i+1], :]
            xfeax = self.avg_pool(feature)
            xfea = self.local_conv_list[i](xfeax)
            local_feat_list.append(xfea)
        part_feature = torch.cat(
            local_feat_list, dim=2).view(batch, -1, self.part)
        part_feature = part_feature.transpose(1, 2).contiguous()
        part_K = self.W_K(part_feature)
        part_Q = self.W_Q(part_feature).transpose(1, 2).contiguous()
        weight_att = torch.bmm(part_K, part_Q)
        attention = torch.softmax(weight_att, dim=2)
        aggreateF = torch.bmm(
            attention, part_feature).transpose(1, 2).contiguous()
        features = []
        for i in range(self.part):
            feature = aggreateF[:, :, i:i +
                                1].expand(batch, self.hidden, IHs[i+1]-IHs[i])
            feature = feature.view(batch, self.hidden, IHs[i+1]-IHs[i], 1)
            features.append(feature)
        features = torch.cat(features, dim=2).expand(batch, self.hidden, H, W)
        final_fea = torch.cat((x, features), dim=1)
        outputs = self.last(final_fea)
        return outputs
