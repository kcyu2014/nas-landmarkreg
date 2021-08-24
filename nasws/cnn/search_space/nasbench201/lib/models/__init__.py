#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/26 下午4:34
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================

import torch
from os import path as osp

__all__ = ['change_key', 'get_cell_based_tiny_net', 'get_search_spaces', 'get_cifar_models', 'get_imagenet_models', \
           'obtain_model', 'obtain_search_model', 'load_net_from_checkpoint', \
           'CellStructure', 'CellArchitectures'
           ]

# useful modules
from ..config_utils import dict2config
from .SharedUtils import change_key
from .cell_searchs import CellStructure, CellArchitectures

# Cell-based NAS Models
def get_cell_based_tiny_net(config):
  group_names = ['DARTS-V1', 'DARTS-V2', 'GDAS', 'SETN', 'ENAS', 'RANDOM']
  from .cell_searchs import nas_super_nets
  if config.name in group_names:
    return nas_super_nets[config.name](config.C, config.N, config.max_nodes, config.num_classes, config.space)
  elif config.name == 'infer.tiny':
    from .cell_infers import TinyNetwork
    return TinyNetwork(config.C, config.N, config.genotype, config.num_classes)
  else:
    raise ValueError('invalid network name : {:}'.format(config.name))


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
def get_search_spaces(xtype, name):
  if xtype == 'cell':
    from .cell_operations import SearchSpaceNames
    return SearchSpaceNames[name]
  else:
    raise ValueError('invalid search-space type is {:}'.format(xtype))


def get_cifar_models(config):
  from .CifarResNet      import CifarResNet
  from .CifarWideResNet  import CifarWideResNet
  
  super_type = getattr(config, 'super_type', 'basic')
  if super_type == 'basic':
    if config.arch == 'resnet':
      return CifarResNet(config.module, config.depth, config.class_num, config.zero_init_residual)
    elif config.arch == 'wideresnet':
      return CifarWideResNet(config.depth, config.wide_factor, config.class_num, config.dropout)
    else:
      raise ValueError('invalid module type : {:}'.format(config.arch))
  elif super_type.startswith('infer'):
    from .shape_infers import InferWidthCifarResNet
    from .shape_infers import InferDepthCifarResNet
    from .shape_infers import InferCifarResNet
    assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
    infer_mode = super_type.split('-')[1]
    if infer_mode == 'width':
      return InferWidthCifarResNet(config.module, config.depth, config.xchannels, config.class_num, config.zero_init_residual)
    elif infer_mode == 'depth':
      return InferDepthCifarResNet(config.module, config.depth, config.xblocks, config.class_num, config.zero_init_residual)
    elif infer_mode == 'shape':
      return InferCifarResNet(config.module, config.depth, config.xblocks, config.xchannels, config.class_num, config.zero_init_residual)
    else:
      raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
  else:
    raise ValueError('invalid super-type : {:}'.format(super_type))


def get_imagenet_models(config):
  super_type = getattr(config, 'super_type', 'basic')
  # NAS searched architecture
  if super_type.startswith('infer'):
    assert len(super_type.split('-')) == 2, 'invalid super_type : {:}'.format(super_type)
    infer_mode = super_type.split('-')[1]
    if infer_mode == 'shape':
      from .shape_infers import InferImagenetResNet
      from .shape_infers import InferMobileNetV2
      if config.arch == 'resnet':
        return InferImagenetResNet(config.block_name, config.layers, config.xblocks, config.xchannels, config.deep_stem, config.class_num, config.zero_init_residual)
      elif config.arch == "MobileNetV2":
        return InferMobileNetV2(config.class_num, config.xchannels, config.xblocks, config.dropout)
      else:
        raise ValueError('invalid arch-mode : {:}'.format(config.arch))
    else:
      raise ValueError('invalid infer-mode : {:}'.format(infer_mode))
  else:
    raise ValueError('invalid super-type : {:}'.format(super_type))


def obtain_model(config):
  if config.dataset == 'cifar':
    return get_cifar_models(config)
  elif config.dataset == 'imagenet':
    return get_imagenet_models(config)
  else:
    raise ValueError('invalid dataset in the model config : {:}'.format(config))


def obtain_search_model(config):
  if config.dataset == 'cifar':
    if config.arch == 'resnet':
      from .shape_searchs import SearchWidthCifarResNet
      from .shape_searchs import SearchDepthCifarResNet
      from .shape_searchs import SearchShapeCifarResNet
      if config.search_mode == 'width':
        return SearchWidthCifarResNet(config.module, config.depth, config.class_num)
      elif config.search_mode == 'depth':
        return SearchDepthCifarResNet(config.module, config.depth, config.class_num)
      elif config.search_mode == 'shape':
        return SearchShapeCifarResNet(config.module, config.depth, config.class_num)
      else: raise ValueError('invalid search mode : {:}'.format(config.search_mode))
    else:
      raise ValueError('invalid arch : {:} for dataset [{:}]'.format(config.arch, config.dataset))
  elif config.dataset == 'imagenet':
    from .shape_searchs import SearchShapeImagenetResNet
    assert config.search_mode == 'shape', 'invalid search-mode : {:}'.format( config.search_mode )
    if config.arch == 'resnet':
      return SearchShapeImagenetResNet(config.block_name, config.layers, config.deep_stem, config.class_num)
    else:
      raise ValueError('invalid model config : {:}'.format(config))
  else:
    raise ValueError('invalid dataset in the model config : {:}'.format(config))


def load_net_from_checkpoint(checkpoint):
  assert osp.isfile(checkpoint), 'checkpoint {:} does not exist'.format(checkpoint)
  checkpoint   = torch.load(checkpoint)
  model_config = dict2config(checkpoint['model-config'], None)
  model        = obtain_model(model_config)
  model.load_state_dict(checkpoint['base-model'])
  return model
