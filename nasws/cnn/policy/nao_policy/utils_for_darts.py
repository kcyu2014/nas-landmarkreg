import copy
import numpy as np
from .utils import NAOParsing
from nasws.cnn.search_space.darts.operations import WSBNOPS
from nasws.cnn.search_space.darts.genotype import PRIMITIVES, Genotype
from nasws.cnn.search_space.darts.darts_search_space import DartsModelSpec
ALLOWED_OPS = PRIMITIVES

DARTS_Node2ArchLength = {
    k: k*2*4 for k in range(2, 5)
}


class NAOParsingDarts(NAOParsing):

    def __init__(self, dataset, args) -> None:
        self.dataset = dataset
        self.args = args
        self.num_nodes = args.num_intermediate_nodes
        self.num_ops = len(PRIMITIVES)

    @staticmethod
    def augmentation(arch):
        split = len(arch) // 2
        num_nodes = len(arch) // 2 // 4
        new_arch = copy.deepcopy(arch)
        for i in range(2):
            rand = np.random.randint(0, num_nodes)
            start = i * split + rand* 4
            end = start + 4
            new_arch[start:end] = new_arch[start + 2: end] + new_arch[start: start+2]
        return new_arch

    def generate_arch(self, n, num_nodes, num_ops=8):
        """ Here we know the architecture num_nodes = num_inter + 2, so we add another 1 """
        # def _get_arch():
        #     arch = []
        #     for i in range(2, num_nodes):
        #         p1 = np.random.randint(0, i)
        #         op1 = np.random.randint(0, num_ops)
        #         p2 = np.random.randint(0, i)
        #         op2 = np.random.randint(0 ,num_ops)
        #         arch.extend([p1, op1, p2, op2])
        #     return arch
        # archs = [_get_arch() + _get_arch() for i in range(n)] #[[[conv],[reduc]]]
        num_nodes = num_nodes or self.num_nodes
        archs = []
        ids = set()
        for _ in range(n):
            while True:
                mid, model_spec = self.dataset.random_topology()
                if mid not in ids:
                    break
            archs.append(self.parse_model_spec_to_arch(model_spec))
            ids.add(mid)
        return archs

    @staticmethod
    def parse_arch_to_model_spec(arch, branch_length=None, B=None):
        # we have two cell
        length = len(arch)
        conv_dag = arch[:length//2]
        reduc_dag = arch[length//2:]
        B = len(conv_dag) // 4
        def _parse_cell(cell):
            # cell[i] == node, cell[i+1] == op_id, reverse in the genotype.
            return [(PRIMITIVES[cell[i+1]], cell[i]) for i in range(0, len(cell), 2)]
        
        g = Genotype(
            normal=_parse_cell(conv_dag), normal_concat=list(range(2, 2+B)),
            reduce=_parse_cell(reduc_dag), reduce_concat=list(range(2, 2+B))
        )
        return DartsModelSpec.from_darts_genotype(g)

    @staticmethod
    def parse_model_spec_to_arch(model_spec):
        """
        Note that, the arch / seq in NAO training, we have , but in genotypes, we have the opposite.
            arch: [node, op ...]
            Geno: [(Op, node), ...]
        """
        arch = []
        g = model_spec.to_darts_genotype()
        for cell in [g.normal, g.reduce]:
            for c in cell:
                arch.extend([c[1], PRIMITIVES.index(c[0])])
        return arch

    # @staticmethod
    # def deserialize_arch(arch):
    #     if arch is None:
    #         return None, None
    #     # assume arch is the format [idex, op ...] where index is in [0, 5] and op in [0, 10]
    #     arch = list(map(int, arch.strip().split()))
    #     return conv_dag, reduc_dag

    # @staticmethod
    # def serialize_arch(arch):
    #     return ' '.join(map(str, arch[0])) + ' '.join(map(str, arch[1]))

    @staticmethod
    def parse_arch_to_seq(arch, branch_length=2, B=4):
        """

        :param arch: when branch_length = 2, arch length = seq length.
        :param branch_length:
        :return: sequence in a very WEIRD way.
        """
        assert branch_length in [2, 3]
        seq = []
        
        def _parse_op(op):
            if op == 0:
                return 7, 12
            if op == 1:
                return 8, 11
            if op == 2:
                return 8, 12
            if op == 3:
                return 9, 11
            if op == 4:
                return 10, 11

        for i in range(B*2): # two cell in one arch
            prev_node1 = arch[4*i]+1
            prev_node2 = arch[4*i+2]+1
            if branch_length == 2:
                op1 = arch[4*i+1] + 2 + B
                op2 = arch[4*i+3] + 2 + B
                seq.extend([prev_node1, op1, prev_node2, op2])
            else:
                op11, op12 = _parse_op(arch[4*i+1])
                op21, op22 = _parse_op(arch[4*i+3])
                seq.extend([prev_node1, op11, op12, prev_node2, op21, op22]) #nopknopk
        return seq

    @staticmethod
    def parse_seq_to_arch(seq, branch_length=2, B=4):
        """
        Why you need this?
        :param seq:
        :param branch_length:
        :return:
        """
        n = len(seq)
        assert branch_length in [2, 3]
        assert n // 2 // (B) // 2 == branch_length
        
        def _parse_arch(arch_seq):
            arch_arch = []
            
            def _recover_op(op1, op2):
                if op1 == 7:
                    return 0
                if op1 == 8:
                    if op2 == 11:
                        return 1
                    if op2 == 12:
                        return 2
                if op1 == 9:
                    return 3
                if op1 == 10:
                    return 4
            if branch_length == 2:
                for i in range(B):
                    p1 = arch_seq[4*i] - 1
                    op1 = arch_seq[4*i+1] - (2 + B)
                    p2 = arch_seq[4*i+2] - 1
                    op2 = arch_seq[4*i+3] - (2 + B)
                    arch_arch.extend([p1, op1, p2, op2])
                return arch_arch
            else:
                for i in range(B):
                    p1 = arch_seq[6*i] - 1
                    op11 = arch_seq[6*i+1]
                    op12 = arch_seq[6*i+2]
                    op1 = _recover_op(op11, op12)
                    p2 = arch_seq[6*i+3] - 1
                    op21 = arch_seq[6*i+4]
                    op22 = arch_seq[6*i+5]
                    op2 = _recover_op(op21, op22)
                    arch_arch.extend([p1, op1, p2, op2])
                return arch_arch
        conv_seq = seq[:n//2]
        reduc_seq = seq[n//2:]
        conv_arch = _parse_arch(conv_seq)
        reduc_arch = _parse_arch(reduc_seq)
        arch = conv_arch + reduc_arch
        return arch
