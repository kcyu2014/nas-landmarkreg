#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/26 下午3:31
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================
import random
from copy import deepcopy

from .lib.config_utils import dict2config
from .lib.models import get_cifar_models, get_cell_based_tiny_net, get_search_spaces, CellStructure
from .lib.models.cell_infers.tiny_network import TinyNetwork as NASBench201NetBackbone


def random_architecture_func_full(max_nodes, op_names):
    """ return a random architecture with only random ops """
    def random_architecture():
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = random.choice(op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return random_architecture


def mutate_arch_func(op_names):
    """
    Computes the architecture for a child of the given parent architecture.
    The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
    """

    def mutate_arch_func(parent_arch):
        child_arch = deepcopy(parent_arch)
        node_id = random.randint(0, len(child_arch.nodes) - 1)
        node_info = list(child_arch.nodes[node_id])
        snode_id = random.randint(0, len(node_info) - 1)
        xop = random.choice(op_names)
        while xop == node_info[snode_id][0]:
            xop = random.choice(op_names)
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple(node_info)
        return child_arch

    return mutate_arch_func


def build_nasbench201_model(num_classes, arch):
    """ return an example nasbench-102 network by random sampling. """

    arch_config = {'channel': 16, 'num_cells': 5}
    net = get_cell_based_tiny_net(dict2config({'name': 'infer.tiny',
                                               'C': arch_config['channel'], 'N': arch_config['num_cells'],
                                               'genotype': arch, 'num_classes': num_classes}
                                              , None)
                                  )
    return net


def example_nasbench201_network_for_cifar():
    """ generate random example. """
    search_space_configs = ['connect-nas', 'aa-nas', 'full']
    # aa-nas is nasbench-102 configuration.
    space = get_search_spaces('cell', search_space_configs[1])
    arch = random_architecture_func_full(3, space)
    return build_nasbench201_model(10, arch)


NasBench201Net = build_nasbench201_model
