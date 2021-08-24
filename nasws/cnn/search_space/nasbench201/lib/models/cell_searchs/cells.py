import math, torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..cell_operations import OPS


class SearchCell(nn.Module):

  def __init__(self, C_in, C_out, stride, max_nodes, op_names):
    super(SearchCell, self).__init__()

    self.op_names  = deepcopy(op_names)
    self.edges     = nn.ModuleDict()
    self.max_nodes = max_nodes
    self.in_dim    = C_in
    self.out_dim   = C_out
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if j == 0:
          xlists = [OPS[op_name](C_in , C_out, stride) for op_name in op_names]
        else:
          xlists = [OPS[op_name](C_in , C_out,      1) for op_name in op_names]
        self.edges[ node_str ] = nn.ModuleList( xlists )
    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)

  def extra_repr(self):
    string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    return string

  def forward(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # GDAS
  def forward_acc(self, inputs, weightss, indexess):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        indexes  = indexess[ self.edge2index[node_str] ].item()
        import pdb; pdb.set_trace() # to-do
        #inter_nodes.append( self.edges[node_str][indexes](nodes[j]) * weights[indexes] )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # joint
  def forward_joint(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
        inter_nodes.append( aggregation )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # uniform random sampling per iteration
  def forward_urs(self, inputs):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      while True: # to avoid select zero for all ops
        sops, has_non_zero = [], False
        for j in range(i):
          node_str   = '{:}<-{:}'.format(i, j)
          candidates = self.edges[node_str]
          select_op  = random.choice(candidates)
          sops.append( select_op )
          if not hasattr(select_op, 'is_zero') or select_op.is_zero == False: has_non_zero=True
        if has_non_zero: break
      inter_nodes = []
      for j, select_op in enumerate(sops):
        inter_nodes.append( select_op(nodes[j]) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # select the argmax
  def forward_select(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( self.edges[node_str][ weights.argmax().item() ]( nodes[j] ) )
        #inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # select the argmax
  def forward_dynamic(self, inputs, structure):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      cur_op_node = structure.nodes[i-1]
      inter_nodes = []
      for op_name, j in cur_op_node:
        node_str = '{:}<-{:}'.format(i, j)
        op_index = self.op_names.index( op_name )
        inter_nodes.append( self.edges[node_str][op_index]( nodes[j] ) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]
