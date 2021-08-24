import torch
import torch.nn as nn
from nasws.cnn.operations import WSBNFull
from nasws.cnn.operations.ofa.ofa_dynamic_ops import DynamicSeparableConv2d

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

BN_OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'max_pool_3x3' : lambda C, stride, bn_kwargs: nn.MaxPool2d(3, stride=stride, padding=1),
  'avg_pool_3x3' : lambda C, stride, bn_kwargs: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'skip_connect' : lambda C, stride, bn_kwargs: Identity() if stride == 1 else FactorizedReduce(C, C, bn_kwargs=bn_kwargs),
  'sep_conv_3x3' : lambda C, stride, bn_kwargs: SepConv(C, C, 3, stride, 1, bn_kwargs=bn_kwargs),
  'sep_conv_5x5' : lambda C, stride, bn_kwargs: SepConv(C, C, 5, stride, 2, bn_kwargs=bn_kwargs),
  'sep_conv_7x7' : lambda C, stride, bn_kwargs: SepConv(C, C, 7, stride, 3, bn_kwargs=bn_kwargs),
  'dil_conv_3x3' : lambda C, stride, bn_kwargs: DilConv(C, C, 3, stride, 2, 2, bn_kwargs=bn_kwargs),
  'dil_conv_5x5' : lambda C, stride, bn_kwargs: DilConv(C, C, 5, stride, 4, 2, bn_kwargs=bn_kwargs),
  'conv_7x1_1x7' : lambda C, stride, bn_kwargs: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, **bn_kwargs)
    ),
}

OFA_OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'max_pool_3x3' : lambda C, stride, bn_kwargs: nn.MaxPool2d(3, stride=stride, padding=1),
  'avg_pool_3x3' : lambda C, stride, bn_kwargs: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'skip_connect' : lambda C, stride, bn_kwargs: Identity() if stride == 1 else FactorizedReduce(C, C, bn_kwargs=bn_kwargs),
  'sep_conv' : lambda C, stride, bn_kwargs: KernelTransformSepConv(C, C, [3,5], stride, 2, bn_kwargs=bn_kwargs),
  'dil_conv' : lambda C, stride, bn_kwargs: KernelTransformDilConv(C, C, [3,5], stride, 4, 2, bn_kwargs=bn_kwargs),
  
}


WSBNOPS = {
  'none' : lambda C, stride, affine, wsbn_in: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine, wsbn_in: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine, wsbn_in: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine, wsbn_in: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine, wsbn_num_inputs=wsbn_in),
  'sep_conv_3x3' : lambda C, stride, affine, wsbn_in: SepConv(C, C, 3, stride, 1, affine=affine, wsbn_num_inputs=wsbn_in),
  'sep_conv_5x5' : lambda C, stride, affine, wsbn_in: SepConv(C, C, 5, stride, 2, affine=affine, wsbn_num_inputs=wsbn_in),
  'dil_conv_3x3' : lambda C, stride, affine, wsbn_in: DilConv(C, C, 3, stride, 2, 2, affine=affine, wsbn_num_inputs=wsbn_in),
  'dil_conv_5x5' : lambda C, stride, affine, wsbn_in: DilConv(C, C, 5, stride, 4, 2, affine=affine, wsbn_num_inputs=wsbn_in),
}

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, wsbn_num_inputs=0, bn_kwargs={}):
    super(ReLUConvBN, self).__init__()
    if not 'affine' in bn_kwargs.keys():
      bn_kwargs['affine'] = affine
    
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, **bn_kwargs) if wsbn_num_inputs == 0 else WSBNFull(wsbn_num_inputs, C_out, **bn_kwargs)
    )
  
  def forward_wsbn(self, x, prev_id):
    return self.forward(x)

  def forward(self, x):
    return self.op(x)


class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, wsbn_num_inputs=0, bn_kwargs={}):
    super(DilConv, self).__init__()
    if not 'affine' in bn_kwargs.keys():
      bn_kwargs['affine'] = affine
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      )
    self.bn = nn.BatchNorm2d(C_out, **bn_kwargs) if wsbn_num_inputs == 0 else WSBNFull(wsbn_num_inputs, C_out, **bn_kwargs) 

  def forward(self, x):
    out = self.op(x)
    return self.bn(out)

  def forward_wsbn(self, x, prev_id):
    out = self.op(x)
    return self.bn(out, prev_id)
  


class KernelTransformDilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size_list, stride, padding, dilation, affine=True, wsbn_num_inputs=0, bn_kwargs={}):
    super(KernelTransformDilConv, self).__init__()
    if not 'affine' in bn_kwargs.keys():
      bn_kwargs['affine'] = affine
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      DynamicSeparableConv2d(C_in, kernel_size_list=kernel_size_list, stride=stride, dilation=dilation),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      )
    self.bn = nn.BatchNorm2d(C_out, **bn_kwargs) if wsbn_num_inputs == 0 else WSBNFull(wsbn_num_inputs, C_out, **bn_kwargs) 
    self.change_kernel_size(3)

  def change_kernel_size(self, kernel_size):
    if kernel_size in self.op[1].kernel_size_list:
      self.op[1].active_kernel_size = kernel_size
      self.op[1].padding = {5:4, 3:2}[kernel_size]

  def forward(self, x):
    out = self.op(x)
    return self.bn(out)

  def forward_wsbn(self, x, prev_id):
    out = self.op(x)
    return self.bn(out, prev_id)
    


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, wsbn_num_inputs=0, bn_kwargs={}):
    super(SepConv, self).__init__()
    if not 'affine' in bn_kwargs.keys():
      bn_kwargs['affine'] = affine
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
    )
    self.op2 = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      )
    self.bn1 = nn.BatchNorm2d(C_in, **bn_kwargs) if wsbn_num_inputs == 0 else WSBNFull(wsbn_num_inputs, C_in, **bn_kwargs)
    self.bn2 = nn.BatchNorm2d(C_out, **bn_kwargs) if wsbn_num_inputs == 0 else WSBNFull(wsbn_num_inputs, C_out, **bn_kwargs)

  def forward(self, x):
    out = self.op(x)
    out = self.bn1(out)
    out = self.op2(out)
    out = self.bn2(out)
    return out
  
  def forward_wsbn(self, x, prev_id):
    out = self.op(x)
    out = self.bn1(out, prev_id)
    out = self.op2(out)
    out = self.bn2(out, prev_id)
    return out
  

class KernelTransformSepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size_list, stride, padding, affine=True, wsbn_num_inputs=0, bn_kwargs={}):
    super(KernelTransformSepConv, self).__init__()
    if not 'affine' in bn_kwargs.keys():
      bn_kwargs['affine'] = affine
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      DynamicSeparableConv2d(C_in, kernel_size_list=kernel_size_list, stride=stride),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
    )
    self.op2 = nn.Sequential(
      nn.ReLU(inplace=False),
      DynamicSeparableConv2d(C_in, kernel_size_list=kernel_size_list, stride=1),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      )
    self.bn1 = nn.BatchNorm2d(C_in, **bn_kwargs) if wsbn_num_inputs == 0 else WSBNFull(wsbn_num_inputs, C_in, **bn_kwargs)
    self.bn2 = nn.BatchNorm2d(C_out, **bn_kwargs) if wsbn_num_inputs == 0 else WSBNFull(wsbn_num_inputs, C_out, **bn_kwargs)
    self.change_kernel_size(3)
  
  def change_kernel_size(self, kernel_size, dilation=None):
    if kernel_size in self.op[1].kernel_size_list:
      self.op[1].active_kernel_size = kernel_size
      self.op2[1].active_kernel_size = kernel_size
      # change dilation accordingly
      dilation = {3: 1, 5:2}[kernel_size]
      padding = {3:1, 5:4}[kernel_size]
      self.op[1].dilation = dilation
      self.op2[1].dilation = dilation
      self.op[1].padding = padding
      self.op2[1].padding = padding

  def forward(self, x):
    out = self.op(x)
    out = self.bn1(out)
    out = self.op2(out)
    out = self.bn2(out)
    return out
  
  def forward_wsbn(self, x, prev_id):
    out = self.op(x)
    out = self.bn1(out, prev_id)
    out = self.op2(out)
    out = self.bn2(out, prev_id)
    return out
    


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x
  
  def forward_wsbn(self, x, prev_id):
    return self.forward(x)


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

  def forward_wsbn(self, x, prev_id):
    return self.forward(x)

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True, wsbn_num_inputs=0, bn_kwargs={}):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    if not 'affine' in bn_kwargs.keys():
      bn_kwargs['affine'] = affine
    self.bn = nn.BatchNorm2d(C_out, **bn_kwargs) if wsbn_num_inputs <= 1  else WSBNFull(wsbn_num_inputs, C_out, **bn_kwargs)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

  def forward_wsbn(self, x, prev_id):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out, prev_id)
    return out

