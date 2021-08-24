"""
This is to adapt the NAS here.

What first should define is the search space

THe idea is basically to replace the FeatureFusionBlock

Let's consider it later.

"""
import torch
import torch.nn as nn
from nni.nas.pytorch.mutables import Mutable
import nni.nas.pytorch.mutables as mutables

from .models.midas_net import MidasNet
from .cell_operations import OPS, UPSAMPLE
from .constants import *


class FusionBlockSearchLayer(mutables.MutableScope):
    # NasBench201 cell.
    def __init__(self, args, features, key):
        super().__init__(key)

        # for the input and output, we do not need to map the features at all. so the operation starts 
        # from the first iteration.
        self.args = args
        self.max_nodes = 2 + args.num_intermediate_nodes
        self.edges = nn.ModuleDict()
        for i in range(2, self.max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                self.edges[node_str] = mutables.LayerChoice(
                    [OPS[p](features, features, 1) for p in PRIMITIVES],
                    key=f'{key}_{node_str}')
        
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)
    
    def forward(self, *inputs):

        if len(inputs) == 1:
            nodes = [torch.zeros_like(inputs[0]), inputs[0]]
        else:
            nodes = [inputs[0], inputs[1]]
        
        for i in range(2, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                inter_nodes.append(self.edges[node_str](nodes[j]))
            nodes.append(sum(inter_nodes))
        output = nodes[-1]

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )
        return output


class FusionBlockSearchWithUpsampleLayer(mutables.MutableScope):
    # NasBench201 cell.
    def __init__(self, args, features, key):
        super().__init__(key)

        # for the input and output, we do not need to map the features at all. so the operation starts 
        # from the first iteration.
        self.args = args
        self.max_nodes = 2 + args.num_intermediate_nodes
        self.edges = nn.ModuleDict()
        for i in range(2, self.max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                self.edges[node_str] = mutables.LayerChoice(
                    [OPS[p](features, features, 1) for p in PRIMITIVES],
                    key=f'{key}_{node_str}')
        self.upsample = mutables.LayerChoice(
            [UPSAMPLE[p](features, features, 2) for p in UPSAMPLE.keys()],
            key=f'{key}_upsample'
        )

        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)
    
    def forward(self, *inputs):

        if len(inputs) == 1:
            nodes = [torch.zeros_like(inputs[0]), inputs[0]]
        else:
            nodes = [inputs[0], inputs[1]]
        
        for i in range(2, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                inter_nodes.append(self.edges[node_str](nodes[j]))
            nodes.append(sum(inter_nodes))
        output = nodes[-1]
        output = self.upsample(output)
        return output


def get_search_space(args):
    return {
        'nasbench201': FusionBlockSearchLayer,
        'nasbench201-upsample': FusionBlockSearchWithUpsampleLayer,
    }[args.search_space]


class MidasNetSearch(MidasNet):

    def __init__(self, path=None, features=256, backbone='resnet50', non_negative=True, args=None):
        # do not pass the path into this, as it will be replaced with midas search branch.
        super().__init__(path=None, features=features, backbone=backbone, non_negative=non_negative)
        self.args = args

        # delete the refinenet branch and re-create as something new.
        del self.scratch.refinenet4
        del self.scratch.refinenet3
        del self.scratch.refinenet2
        del self.scratch.refinenet1

        SpaceLayer = get_search_space(args)

        # replace these layers to searchable ones. LOLLOLL
        self.scratch.refinenet1 = SpaceLayer(args, features, 'fusion1')
        self.scratch.refinenet2 = SpaceLayer(args, features, 'fusion2')
        self.scratch.refinenet3 = SpaceLayer(args, features, 'fusion3')
        self.scratch.refinenet4 = SpaceLayer(args, features, 'fusion4')

        if path:
            print("Loading weights to MidasNetSearch: ", path)
            self.load(path)


    def apply_arch(self, json_file):
        pass