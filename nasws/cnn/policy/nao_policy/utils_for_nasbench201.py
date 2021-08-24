from os import stat
from .utils import NAOParsing
from nasws.cnn.search_space.nasbench201.lib.models import CellStructure, get_search_spaces

import IPython

# only support node 4 case, because the search space is too small otherwise
NASBENCH201_ArchLength2Node = {6: 4, 3: 3, 1:2}
NASBENCH201_Node2ArchLength = {v: k for k, v in NASBENCH201_ArchLength2Node.items()}

ALLOWED_OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

def define_num_nodes_from_arch(arch):
    return NASBENCH201_ArchLength2Node[len(arch)]


class NAOParsingNASBench201(NAOParsing):
    """
        NASBench201 support
        
            ModelSpec - CellStructure
        
        As NB201 is full connection, so we just need to search for the operation on each edge, i.e. 
        this will be similar to ProxylessNAS parsing

    """
    num_ops = 5
    num_nodes = None

    def __init__(self, nasbench, args=None) -> None:
        self.nasbench = nasbench
        self.args = args
        self.num_nodes = self.args.num_intermediate_nodes

    # nonstatic because need to check
    def generate_arch(self, n, num_nodes=None, num_ops=5):
        num_nodes = num_nodes or self.num_nodes
        archs = []
        ids = set()
        for _ in range(n):
            while True:
                mid, model_spec = self.nasbench.random_topology()
                if model_spec.node_num == self.num_nodes:
                    break
            archs.append(self.parse_model_spec_to_arch(model_spec))
            ids.add(mid)
        return archs
    
    @staticmethod
    def parse_model_spec_to_arch(model_spec):
        arch = []
        for node in model_spec.nodes:
            _arch = [0] * len(node)
            for c in node:
                _arch[c[1]] = ALLOWED_OPS.index(c[0])
            arch.extend(_arch)
        return arch

    @staticmethod
    def parse_arch_to_model_spec(arch, B=None):
        B = define_num_nodes_from_arch(arch)
        genotype = []
        offset = 0
        for i in range(B-1):
            genotype.append([ (ALLOWED_OPS[arch[offset + k]], k) for k in range(i+1)])
            offset += i + 1
        return CellStructure(genotype)

    @staticmethod
    def parse_arch_to_seq(arch, branch_length=None, B=None):
        """
        Simply add 1 
        :param cell:
        :param branch_length:
        :param B:     So, B is number of intermeidate layer, and it is shifted by 2 in NB101 case.
        :return:
        """
        return [a + 1 for a in arch]

    @staticmethod
    def parse_seq_to_arch(seq, branch_length=None, B=None):
        """ same as arch2seq, note that B is """
        return [s - 1 for s in seq]
        