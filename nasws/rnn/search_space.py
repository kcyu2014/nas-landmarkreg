
import collections
import math
# import search_configs as configs
from genotypes import PRIMITIVES, Genotype


class SearchSpace:

    def __init__(self, args):

        self.args = args
        self.intermediate_nodes = self.args.num_intermediate_nodes
        self.operations = [op for op in PRIMITIVES if op is not 'none']
        self.num_operations = self.args.num_operations
        self.num_solutions = math.factorial(self.intermediate_nodes) * (self.num_operations ** self.intermediate_nodes)

    def genotype_from_id(self, genotype_id, num_nodes=None):

        if num_nodes is None:
            num_nodes = self.intermediate_nodes

        # used for the leaf nodes logic
        concat = set([elem for elem in range(1, self.intermediate_nodes + 1)])

        gene = [(self.operations[0], 0) for _ in range(num_nodes)]

        current_div_result = genotype_id
        current_node_id = num_nodes - 1

        while current_div_result > 0:
            # print(current_div_result, current_node_id, self.num_operations)
            current_div_result, prec_op_id = divmod(current_div_result, ((current_node_id + 1) * self.num_operations))

            prec_node_id, operation_id = divmod(prec_op_id, self.num_operations)

            if prec_node_id in concat:
                concat.remove(prec_node_id)

            # updating the edge for the current node slot with the new ids
            gene[current_node_id] = (self.operations[operation_id], prec_node_id)

            # updating to the next node id of the genotype slot, from bottom to top
            current_node_id -= 1

        if self.args.use_avg_leaf:
            genotype = Genotype(recurrent=gene, concat=sorted(list(concat)))
        else:
            genotype = Genotype(recurrent=gene, concat=range(num_nodes + 1)[-num_nodes:])

        return genotype

    def genotype_id_from_geno(self, genotype):

        previous_slot = 0
        num_operations = len(self.operations)

        for index, gene in enumerate(genotype.recurrent):
            operation = gene[0]

            possible_relative_slots = (index + 1) * num_operations

            operation_id = self.operations.index(operation)

            prev_node_id = gene[1]

            relative_slot = num_operations * prev_node_id + operation_id

            previous_slot = relative_slot + (previous_slot * possible_relative_slots)

        return previous_slot



'''
args = configs.parser.parse_args()
search_space = SearchSpace(args)

genotype = Genotype(recurrent=[('sigmoid', 0), ('identity', 1), ('tanh', 0)], concat=range(1, 4))
genotype_id = search_space.genotype_id_from_geno(genotype)
print(genotype_id)
print(search_space.genotype_from_id(287))
'''


def dag_to_genotype(dag, num_blocks):
    """
    For ENAS RNN to DARTS style. String based to vector based.
    :param dag:
    :param num_blocks:
    :return:
    """
    # keys = []
    ids = num_blocks * [-1,]
    funcs = num_blocks * [None,]

    leaf_node_ids = []
    q = collections.deque()
    q.append(-1)
    while True:
        if len(q) == 0:
            break

        node_id = q.popleft()
        nodes = dag[node_id]

        for next_node in nodes:
            next_id = next_node.id
            if next_id == num_blocks:
                leaf_node_ids.append(node_id)
                assert len(nodes) == 1, ('parent of leaf node should have '
                                         'only one child')
                continue
            ids[next_id] = node_id
            funcs[next_id] = next_node.name.lower()
            q.append(next_id)

    return Genotype([(f, i + 1) for i,f in zip(ids, funcs)], [i + 1 for i in sorted(leaf_node_ids)])


class ENASSearchSpace(SearchSpace):
    # add a wrapper from genotype to DAG

    def genotype_to_dag(self, geno):
        pass

    def dag_to_geno(self, dag):
        return dag_to_genotype(dag, self.args.num_blocks)




