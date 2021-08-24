import copy
import logging
from collections import deque, namedtuple
import IPython
import numpy as np
import random
import utils
from nasws.cnn.search_space import graph_util


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True


class CNNSearchSpace:

    #### Property ####
    sample_step_for_evaluation = 1
    top_K_complete_evaluate = 200
    landmark_num_archs = 10
    evaluate_ids = None  # stores the pool of evaluate ids. Fixed after initialized.
    evaluate_model_spec_ids = None  # Current pool of ids, can change from time to time.
    num_classes = None
    def __repr__(self):
        return self.__class__.__name__ + ": num archs " + str(self.num_architectures)

    @property
    def topologies(self):
        return self._model_specs

    @property
    def model_specs(self):
        return self._model_specs

    @property
    def hashs(self):
        return self._hashs

    @property
    def num_architectures(self):
        return len(self._hashs)

    @property
    def rank_by_mid(self):
        """ return the rank of given model spec id, but usually, we use rank as id. """
        if self._ranks is None:
            self._ranks = list(range(len(self.hashs)))
        return self._ranks

    def __init__(self, args):
        # only take this as input.
        self.args = args
        self.topology_fn = None  # Store the topology function.
        self.model_fn = None     # store the model creation function
        self._hashs = None
        self._model_specs = None
        self._ranks = None
        self._landmark_ids = None
        self.dataset = None
        self._perfs = None
        self._losses = None

    def _construct_search_space(self):
        """ constructing search space based on dataset. """
        _hashs = []
        _model_specs = []
        _perfs = []
        _losses = []
        # self._ranks = None
        # self._landmark_id = None

        # load the dataset
        for h, s in self.dataset.hash_dict.items():
            _hashs.append(h)
            _model_specs.append(s)
            _perfs.append(self.dataset.query_perf(s))
            _losses.append(self.dataset.query_loss(s))

        sorted_index = np.argsort(_perfs).tolist()
        if '_subsample' in self.args.search_space:
            logging.info(f'{self.args.search_space} using subsample space {self.args.num_archs_subspace}')
            # reduce the search space to a small number
            if self.args.num_archs_subspace > 0:
                sorted_index = random.sample(sorted_index, k=self.args.num_archs_subspace)
                sorted_index.sort()
            else:
                logging.info('Using FULL space.')
            # IPython.embed(header='check the sorted index sampling and try to save')
            # save this hash to the exp folder... otherwise it is impossible to retrieve the model during test
        
        self._hashs = [_hashs[i] for i in sorted_index]
        self._model_specs = [_model_specs[i] for i in sorted_index]
        self._perfs = [_perfs[i] for i in sorted_index]
        self._losses = [_losses[i] for i in sorted_index]
        self._ranks = list(range(len(self._hashs)))

        logging.info(f"{self.__class__.__name__} loaded. total trained model points {self.num_architectures}")
        # again, sort by the performance.
        self.landmark_num_archs = self.args.landmark_num_archs
        self.reset_landmark_topologies()
        self.initialize_evaluate_pool()

    def initialize_evaluate_pool(self):
        # Process evaluation nodes
        self.evaluate_ids = [i for i in range(0, self.num_architectures, self.sample_step_for_evaluation)]
        # remove the landmark id from eval_ids
        if self.landmark_num_archs > 0:
            # IPython.embed()
            self.evaluate_ids = list(set(self.evaluate_ids) - set(self._landmark_ids))

        self.evaluate_model_spec_ids = deque(self.evaluate_ids)
        if len(self.evaluate_model_spec_ids) > self.top_K_complete_evaluate:
            self.evaluate_model_spec_ids = deque(sorted(
                np.random.choice(self.evaluate_model_spec_ids, self.top_K_complete_evaluate, replace=False).tolist()))

    def evaluate_model_spec_id_pool(self):
        if len(self.evaluate_model_spec_ids) > self.top_K_complete_evaluate:
            self.evaluate_model_spec_ids = deque(sorted(
                np.random.choice(self.evaluate_model_spec_ids, self.top_K_complete_evaluate,
                                 replace=True).tolist()))
        return self.evaluate_model_spec_ids

    def eval_model_spec_id_append(self, mid):
        if mid in self.evaluate_model_spec_ids:
            self.evaluate_model_spec_ids.remove(mid)

        if len(self.evaluate_model_spec_ids) >= self.top_K_complete_evaluate:
            old_arch = self.evaluate_model_spec_ids.pop()
            logging.debug("Pop arch {} from pool".format(old_arch))
        self.evaluate_model_spec_ids.append(mid)
        logging.debug("Push arch {} to pool".format(mid))
    
    def eval_model_spec_id_rank(self, ids, perfs):
        """
        Rank the evaluate id pools by the performance.

        :param ids:
        :param perfs:
        :return: None
        """
        # rank the thing, make sure the pop-left will eliminate the poor performed archs.
        old_archs_sorted_indices = np.argsort(perfs)[::-1]
        rank_ids = [ids[i] for i in old_archs_sorted_indices]
        if len(rank_ids) > self.top_K_complete_evaluate:
            rank_ids = rank_ids[:self.top_K_complete_evaluate]
        self.evaluate_model_spec_ids = deque(rank_ids)

    def random_topology(self):
        """
        Naive random sampling method.
        :return: id, spec
        """
        rand_spec_id = self.random_ids(1)[0]
        rand_spec = self.topologies[rand_spec_id]
        return rand_spec_id, rand_spec

    def random_topology_random_nas(self):
        raise NotImplementedError('To implement in each individual search spaces.')

    def mutate_topology(self, old_spec, mutation_rate=1.0):
        # Mutate the thing.
        raise NotImplementedError("mutate_topology should be implemented in sub-class for each space.")

    def random_ids(self, number):
        return sorted(np.random.choice(np.arange(0, self.num_architectures), number, replace=False).tolist())

    def random_eval_ids(self, number):
        """
        Random a couple of eval ids from the Evaluation pool, but not the current ids.
        :param number:
        :return:
        """
        return sorted(np.random.choice(self.evaluate_ids, min(number, len(self.evaluate_ids)),
                                       replace=False).tolist())

    def hash_by_id(self, i):
        """ return model hash by id """
        return self.hashs[i]

    def topology_by_id(self, i):
        """ return topoligy by id """
        return self.topologies[i]
    
    def topology_to_id(self, spec):
        h = self.serialize_model_spec(spec)
        try:
            return self.hashs.index(h)
        except:
            return None

    def validate_model_indices(self, valid_queue_length, sampling=None):
        """
        Process for validation step during training supernet.
        This step for the current being is a random selection without any prior knowledge.
        Possible to support another version.

        :param valid_queue_length:
        :return: valid_model_pool
        """
        nb_models = self.num_architectures
        nb_batch_per_model = max(valid_queue_length // nb_models, 1)
        if sampling is None:
            valid_model_order = np.random.choice(range(nb_models), nb_models, False)
        else:
            raise NotImplementedError("not yet supported. to add in future.")

        if nb_models > valid_queue_length:
            valid_model_order = valid_model_order[:valid_queue_length]
            nb_models = valid_queue_length
        return nb_batch_per_model, nb_models, valid_model_order

    def replace_eval_ids_by_random(self, number):
        """ Random a subset and replace the bottom performed architectures. """
        replace_number = 0
        rand_eval_ids = self.random_eval_ids(number)
        for eid in rand_eval_ids:
            if eid not in self.evaluate_model_spec_ids:
                self.eval_model_spec_id_append(eid)
                replace_number += 1
        return replace_number

    def process_archname_by_id(self, arch):
        # arch is mid
        return f"{arch}, {self.hashs[arch]}"
    
    def generate_new_arch(self, number):
        """
        Return the id or not.
        :param number:
        :return:
        """
        archs = []
        for _ in range(number):
            _, m = self.random_topology()
            archs.append(m)
        return archs

    @property
    def landmark_topologies(self):
        """
        This is used a landmark ids.
        Make sure this is not aligned with initial sampled pool
        :return: topology_id, topology.
        """
        if self.landmark_num_archs == 0:
            return [], []
        
        # by default, compute the landmark on the request.
        if self._landmark_ids is None:
            if self.args.landmark_sample_method == 'fixed':
                step = self.num_architectures // self.landmark_num_archs
                start = self.num_architectures - (self.landmark_num_archs - 1) * step - 1
                ids = list(range(start, self.num_architectures, step))
                assert len(ids) == self.landmark_num_archs, print(len(ids), self.landmark_num_archs)

            elif self.args.landmark_sample_method == 'random':
                ids = self.random_ids(self.landmark_num_archs)
            else:
                raise NotImplementedError("sample method not yet supported.")

            self._landmark_ids = ids
        else:
            ids = self._landmark_ids
        model_specs = [self.topologies[i] for i in ids]
        return ids, model_specs
    
    @landmark_topologies.setter
    def landmark_topologies(self, topologies):
        """ set the landmark topologies """
        self._landmark_ids = None
        self._landmark_weights = None
        logging.info(f'Reset landmark topologies to {topologies}')
        _ids = []
        for t in topologies:
            if isinstance(t, int) and t < len(self.topologies):
                # as ID
                _ids.append(t)
            else:
                # as Topology, translate to ID
                index = self.topology_to_id(t)
                if index:
                    _ids.append(index)
        self._landmark_ids = _ids

    @property
    def landmark_weights(self):
        if self._landmark_weights:
            return self._landmark_weights
        
        ids = self._landmark_ids
        # process landmark weights.
        if self.args.landmark_loss_weighted == 'embed':
            weights = np.array(self.query_gt_perfs(ids), dtype=np.float)
            # normalize this to [0, 1] to use difference, to make sure.
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
            self._landmark_weights = weights.tolist()
            # IPython.embed(header='test here if the weights are correctly normalized to 0 - 1.')
        elif self.args.landmark_loss_weighted == 'infinite':
            weights = np.array(self.query_gt_perfs(ids), dtype=np.float)
            self._landmark_weights = weights.tolist()
        return self._landmark_weights

    def reset_landmark_topologies(self):
        """ Reset the landmark ids and weights to train ranking loss. """
        self._landmark_ids = None
        self._landmark_weights = None
        return self.landmark_topologies

    def query_gt_perfs(self, model_ids):
        if all(map(lambda x: isinstance(x, int), model_ids)):
            return [self._perfs[i] for i in model_ids]
        return None
    
    def query_gt_loss(self, model_ids):
        return [self._losses[i] for i in model_ids]

    @staticmethod
    def change_model_spec(model, spec):
        """Change model spec 

        Parameters
        ----------
        model : Supernet or DataParallel(Supernet)
            Supernet to alter
        spec : ModelSpec 
            Each individual search space will have its own spec

        Raises
        ------
        NotImplementedError
            To be implemented in the sub-class.
        """
        raise NotImplementedError('Change model spec should be implemented by the search space to mutate the supernet')
    
    @staticmethod
    def module_forward_fn(model, input, target, criterion):
        """
        Wrapper for other stuff:
        
            outputs = model(input)
            loss = criterion(outputs, target)
        
        """
        raise NotImplementedError('To be implemented in each individual classes...')

    """ Utility """
    def save_search_space_snapsnot(self, path):
        # this will save the entire search space as a snapshot just for reference
        keys = ['hashs', 'model_specs', 'ranks', 'landmark_ids', 'perfs', 'losses']
        utils.save_json(
            {k: getattr(self, '_' + k) for k in keys}, 
            path
        )

    def load_search_space_snapshot(self, path):
        d = utils.load_json(path)
        for k in ['hashs', 'model_specs', 'ranks', 'landmark_ids', 'perfs', 'losses']:
            setattr(self, f'_{k}', d[k])
    
    def check_valid(self, model_spec):
        raise NotImplementedError()

    def serialize_model_spec(self, model_spec):
        return model_spec.hash_spec()

    def deserialize_model_spec(self, spec_str):
        if spec_str in self.hashs:
            ind = self.hashs.index(spec_str)
            return self.topology_by_id(ind)
        return None
    

class OpSpecTemplate(object):
    """ Type for ModelSpecTemplate.ops
    Support the generic modelspec creation.
        Example:
             List:
    """

    typename = 'TemplateToken'
    op_choices = {
        'op1_name': [],
        'op2_name': []
    }
    _type = 'List' # only list choice. This will correspond with the adjacent matrix.
    auxilary_node = {}
    output_node_number = 1

    def __init__(self, ops, ops_type):
        self.ops = ops
        self._type = ops_type

    @classmethod
    def build_op_from_list(cls, input_list):
        raise NotImplementedError("To add for list type")

    @classmethod
    def build_op_from_matrix(cls, input_matrix):
        raise NotImplementedError("To add for matrix type")

    def __repr__(self):
        return f"Type: {self._type} \n Ops: {self.ops}"

    def __len__(self):
        if self._type.lower() == 'list':
            return len(self.ops)
        else:
            return len(self.ops.shape[0])

    def __getitem__(self, item):
        return self.ops.__getitem__(item)

    def __delitem__(self, key):
        del self.ops[key]

    @property
    def classmethod_token(self):
        """ return the classmethod to to create a token. """
        return namedtuple(self.typename, list(self.op_choices.keys()))

    @property
    def labeling(self):
        """ labeling to compute the hash """
        raise NotImplementedError("should be implemented by each spec.")

    def check_consistency(self, matrix, single_op_per_node=False):
        """
        Check the ops is consistent with matrix.
        :param matrix:
        :param single_op_per_node: True to support NASbench like, only one operation per node,
                                   otherwise, a list or tuple
        :return:
        """
        output_node_number = 1
        offset = len(self.auxilary_node.keys()) - self.output_node_number
        # number of connections
        nb_connection = np.sum(matrix, axis=0)
        is_valid = True

        # remove the last output node. because it is not relavent.
        for ind, connect in enumerate(nb_connection.tolist()[offset:-self.output_node_number]):
            if isinstance(self.ops[ind], (list, tuple)):
                if connect == 0:
                    pass
                elif connect == 1:
                    if len(self.ops[ind]) != 1:
                        logging.debug(f"Ops checking consistency error: "
                                      f"connect == 1 but ops at {ind} number (== {len(self.ops[ind])}) != 1")
                        is_valid = False
                else:   # connect > 1
                    len_ops = 1 if isinstance(self.ops[ind], str) else len(self.ops[ind])
                    if len_ops != connect:
                        logging.debug(f"Ops checking consistency error: node {ind} got {connect}"
                                      f"connection but  {len_ops}  ops. Details ops {self.ops[ind]}")
                        is_valid = False
            else:
                if connect == 1 and not single_op_per_node:
                    logging.debug(f"Ops checking consistency error: node {ind} got {connect} == 1, "
                                  f"but should not be single. {self.ops[ind]}")
                    is_valid = False

        return is_valid


class CellSpecTemplate(object):

    matrix = None       # adjacent matrix.    dimension is equal to number of total node.
    ops = None          # Ops matrix or list. dimension the same as matrix
    node_name = None    # name of each node. for better understanding.
    valid_spec = True   # valid spec

    def __init__(self, matrix, ops: OpSpecTemplate, data_format='channels_last',
                 num_cell=1):
        """Initialize the module spec.

        Args:
          matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
          ops: V-length list of labels for the base ops used. The first and last
            elements are ignored because they are the input and output vertices
            which have no operations. The elements are retained to keep consistent
            indexing.
          data_format: channels_last or channels_first.

        Raises:
          ValueError: invalid matrix or ops
        """

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('matrix must be square')
        if shape[0] != len(ops):
            raise ValueError('length of ops must match matrix dimensions')
        if not is_upper_triangular(matrix):
            raise ValueError('matrix must be upper triangular')

        # Both the original and pruned matrices are deep copies of the matrix and
        # ops so any changes to those after initialization are not recognized by the
        # spec.
        self.original_matrix = copy.deepcopy(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(matrix)
        self.ops = copy.deepcopy(ops)
        self.valid_spec = True
        self._prune()
        # check ops consistency
        try:
            self.valid_spec = self.ops.check_consistency(matrix)
        except AttributeError:
            self.valid_spec = False
        self.data_format = data_format

    """ Auxilary functions """
    def _prune(self):
        # prune the unnecessary part of a graph, and re-order them to be topologically identical.
        """Prune the extraneous parts of the graph.

        General procedure:
          1) Remove parts of graph not connected to input.
          2) Remove parts of graph not connected to output.
          3) Reorder the vertices so that they are consecutive after steps 1 and 2.

        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        num_vertices = np.shape(self.original_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            self.valid_spec = False
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]

    def hash_spec(self, labeling=None):
        """Computes the isomorphism-invariant graph hash of this spec.

        Args:
          labeling: list of label that match the matrix size. should be in numbers

        Returns:
          MD5 hash of this spec which can be used to query the dataset.
        """
        # Invert the operations back to integer label indices used in graph gen.
        labeling = self.ops.labeling
        return graph_util.hash_module(self.matrix, labeling)

    def visualize(self):
        """Creates a dot graph. Can be visualized in colab directly."""
        num_vertices = np.shape(self.matrix)[0]
        try:
            import graphviz
            g = graphviz.Digraph()
            g.node(str(0), 'input')
            for v in range(1, num_vertices - 1):
                g.node(str(v), self.ops[v])
            g.node(str(num_vertices - 1), 'output')

            for src in range(num_vertices - 1):
                for dst in range(src + 1, num_vertices):
                    if self.matrix[src, dst]:
                        g.edge(str(src), str(dst))

            return g
        except ImportError as e:
            print(e)

    def __str__(self):
        return 'Adjacent matrix: {} \n' \
               'Ops: {}\n'.format(self.matrix.tolist(), self.ops)

    def new(self):
        from copy import copy
        return self.__class__(self.matrix.copy(), copy(self.ops))

    def resume_original(self):
        """
        For NASbench with supernet training, we need to keep topology the
        same while keeping the hash sepc correct.
        :return:
        """
        # These two used to compute hash.
        self.matrix_for_hash = self.matrix
        self.ops_for_hash = self.ops

        self.matrix = copy.deepcopy(self.original_matrix)
        self.ops = copy.deepcopy(self.original_ops)

    """ add for topology output usage """
    def to_dag(self):
        """ return a NetworkX structure """
        raise NotImplementedError("to do later.")

    def to_str_token(self):
        """ return string based token, for RL """
        raise NotImplementedError("to do later")

    def to_dag_enas(self):
        """ as title. """
        raise NotImplementedError("to do later.")


__all__ = ['CNNSearchSpace', 'CellSpecTemplate', 'OpSpecTemplate']
