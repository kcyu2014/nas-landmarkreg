import copy
import numpy as np
from .utils import NAOParsing
from nasws.cnn.search_space.nasbench101.sampler import random_spec, obtain_full_model_spec
from nasws.cnn.search_space.nasbench101.nasbench_api_v2 import ModelSpec_v2

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NASBENCH101_ArchLength2Node = {
        1:  2, 
        4:  3,
        8:  4,
        13: 5,
        19: 6,
        26: 7
            }

NASBENCH_Node2ArchLength = {v: k for k, v in NASBENCH101_ArchLength2Node.items()}


def compute_op_ids(num_nodes):
    op_ids = []
    offset = 0
    for i in range(1, num_nodes- 1):
        offset += (i + 1)
        op_ids.append(offset-1)
    return op_ids


def define_num_nodes_from_arch(arch):
    return NASBENCH101_ArchLength2Node[len(arch)]


class NASBench101NAOParsing(NAOParsing):
    """
        Full NASBench 101 support, this will be a chance to revisit nasbench experiments with NAO
    
    """
    num_ops = 3
    num_nodes = None

    def __init__(self, nasbench, args=None) -> None:
        self.nasbench = nasbench
        self.args = args
        self.num_nodes = self.args.num_intermediate_nodes + 2

    # nonstatic because need to check
    def generate_arch(self, n, num_nodes=None, num_ops=3):
        # ensure that we only accept a good model architecture
        num_nodes = num_nodes or self.num_nodes
        archs = []
        ids = set()
        for _ in range(n):
            while True:
                mid, model_spec = self.nasbench.random_topology()
                if len(model_spec.ops) == num_nodes and (mid not in ids):
                    break
            archs.append(self.parse_model_spec_to_arch(model_spec))
            ids.add(mid)
        return archs

    def parse_model_spec_to_arch(self, model_spec):
        matrix = model_spec.matrix
        ops = model_spec.ops
        if len(ops) < self.num_nodes:
            raise ValueError('WRONG! model spec here must match predefined num_intermediate_nodes.'
                             'Otherwise the NAO arch will change and cause issue... ')
        # for each node, we just parsed the sequenc one by one.
        # length of this spec will be
        arch = []
        for i in range(1, len(ops)):
            # connection of i node to previous
            for c in matrix[:i, i]:
                arch.append(0 if c == 0 else 1)
            if i == len(ops) - 1:
                continue
            arch.append(ALLOWED_OPS.index(ops[i]))
        return arch

    @staticmethod
    def parse_arch_to_model_spec(arch, B=None):
        B = B or define_num_nodes_from_arch(arch)
        matrix = np.zeros((B, B)).astype(np.int64)
        oids = compute_op_ids(B)
        cids = [-1,] + oids + [len(arch)]
        for i in range(1, B):
            # print('assign arch[cids[i-1]: cids[i]]', arch[cids[i-1]+1: cids[i]])
            matrix[:i, i] = arch[cids[i-1]+1: cids[i]]
        ops = [INPUT] + [ALLOWED_OPS[arch[oid]] for oid in oids] + [OUTPUT]
        return ModelSpec_v2(matrix, ops)

    @staticmethod
    def parse_arch_to_seq(arch, branch_length=None, B=None):
        """
        Hard-coded mapping of architecture to this so called sequence.
        Rule:
            Node indicator will map from 0,1 to 1, 2?
        
        TODO This is very tricky, need to make sure why...

        :param cell:
        :param branch_length:
        :param B:     So, B is number of intermeidate layer, and it is shifted by 2 in NB101 case.
        :return:
        """
        B = B + 2 or define_num_nodes_from_arch(arch)
        seq = copy.deepcopy(arch)
        oids = compute_op_ids(B)
        for i in range(len(seq)):
            if i in oids:
                seq[i] += 3
            else:
                seq[i] += 1
        return seq

    @staticmethod
    def parse_seq_to_arch(seq, branch_length=None, B=None):
        """ same as arch2seq, note that B is """
        B = B + 2 or define_num_nodes_from_arch(seq)
        arch = copy.deepcopy(seq)
        oids = compute_op_ids(B)
        for i in range(len(arch)):
            if i in oids:
                arch[i] -= 3
            else:
                arch[i] -= 1
        return arch


class NASBench101NAOParsing_v1(NAOParsing):
    @staticmethod
    def generate_arch(n, num_nodes, num_ops=3):
        def _get_arch():
            arch = []
            for i in range(1, num_nodes-1):
                p1 = np.random.randint(0, i)
                op1 = np.random.randint(0, num_ops)
                arch.extend([p1, op1])
            return arch
        archs = [_get_arch() for i in range(n)] #[archs,]
        return archs

    @staticmethod
    def build_dag(arch):
        if arch is None:
            return None
        # assume arch is the format [idex, op ...] where index is in [0, 5] and op in [0, 10]
        arch = list(map(int, arch.strip().split()))
        # length = len(arch)
        return arch

    @staticmethod
    def parse_arch_to_model_spec(arch, B=None):
        matrix, ops = NASBench101NAOParsing_v1.parse_arch_to_model_spec_matrix_op(arch, len(arch) // 2)
        model_spec = ModelSpec_v2(matrix, ops)
        return model_spec

    @staticmethod
    def parse_arch_to_model_spec_matrix_op(cell, B=5):
        matrix = np.zeros((B+2, B+2))
        ops = [INPUT,]
        is_output = [True,] * (B+1)
        for i in range(B):
            prev_node1 = cell[2*i] # O as input.
            op = ALLOWED_OPS[cell[2*i + 1]]
            ops.append(op)

            is_output[prev_node1] = False
            curr_node = i + 1
            matrix[prev_node1][curr_node] = 1
        # process output
        for input_node, connect_to_output in enumerate(is_output):
            matrix[input_node][B+1] = 1 if connect_to_output else 0
        matrix = matrix.astype(np.int)
        ops.append(OUTPUT)
        return matrix, ops
    
    # @staticmethod
    # def parse_model_spec_to_arch(model_spec):
    #     arch = []
    #     for i in range(1, len(model_spec.ops) - 1):

    @staticmethod
    def parse_arch_to_seq(cell, branch_length=2, B=5):
        """
        Hard-coded mapping of architecture to this so called sequence.

        :param cell:
        :param branch_length:
        :param B:     So, B is number of intermeidate layer.
        :return:
        """
        assert branch_length in [2,]
        seq = []
        
        # For Arch [], the only purpose is to encode [0-B] for the prev node and [B+1, ... B+ num_ops]
        for i in range(B):
            prev_node1 = cell[2*i] + 1
            op1 = cell[2*i+1] + B + 1
            seq.extend([prev_node1, op1])
        return seq

    @staticmethod
    def parse_seq_to_arch(seq, branch_length=2, B=5):
        n = len(seq)
        assert branch_length in [2,]
        assert n // B == branch_length
        
        def _parse_cell(cell_seq):
            cell_arch = []
            for i in range(B):
                p1 = cell_seq[2*i] - 1
                op1 = cell_seq[2*i+1] - (B+2)
                cell_arch.extend([p1, op1])
            return cell_arch
            
        conv_seq = seq
        conv_arch = _parse_cell(conv_seq)
        return conv_arch
