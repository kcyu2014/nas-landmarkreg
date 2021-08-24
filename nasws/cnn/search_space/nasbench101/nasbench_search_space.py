"""
2019.08.19, testing the first nasbench search space.
I should finish this within a weekend and should deploy this as soon as possible.
"""
import random
from copy import copy, deepcopy
from functools import partial

import itertools
import logging
import os
from collections import OrderedDict, deque, namedtuple

import IPython

from nasws.cnn.search_space.nasbench101.nasbench_api_v2 import NASBench_v2, ModelSpec_v2
from nasws.cnn.search_space.nasbench101.sampler import obtain_full_model_spec, obtain_random_spec
from nasws.cnn.search_space.search_space import CNNSearchSpace
import numpy as np
from .util import change_model_spec, nasbench_model_forward
from .model import NasBenchNet
from .model_search import NasBenchNetSearch
from nasws.cnn.search_space.nasbench101.model_builder import compute_vertex_channels


class NASbenchSearchSpace(CNNSearchSpace):
    """NASbench Search Space used in the entire search code

    Version 1.0 Finalized, support everything we need.
    Version 1.1 Move the majority of the functions into the CNN Search Space to be reused for other classes.

    """

    # override some meta-variables
    sample_step_for_evaluation = 1
    top_K_complete_evaluate = 200
    landmark_num_archs = 10
    evaluate_ids = None             # stores the pool of evaluate ids. Fixed after initialized.
    evaluate_model_spec_ids = None  # Current pool of ids, can change from time to time.
    num_classes = 10
    
    def __init__(self, args, full_dataset=False):
        """
        Initialize the search space.
        :param args:
        """
        # self.args = args
        super(NASbenchSearchSpace, self).__init__(args)
        self.topology_fn = NasBenchNet
        self.model_fn = NasBenchNetSearch
        self.sample_step_for_evaluation = 1 if not args.debug else 30
        # read nasbench related configs.
        args.model_spec = obtain_full_model_spec(args.num_intermediate_nodes + 2)
        v = self.args.num_intermediate_nodes + 2
        self.nasbench = NASBench_v2(os.path.join(self.args.data, 'nasbench/nasbench_only108.tfrecord'),
                                    config=f'v{v}_e9_op3', only_hash=not full_dataset)
        self.dataset = self.nasbench
        self._construct_search_space()
        self.initialize_evaluate_pool()

    def _construct_search_space(self):
        self.nasbench_hashs, self.nasbench_model_specs = self.nasbench.model_hash_rank(full_spec=True)
        self._ranks = [i for i in range(0, len(self.nasbench_hashs))]
        self.available_ops = self.nasbench.available_ops
        self.landmark_num_archs = self.args.landmark_num_archs
        self._landmark_ids = None
        self._template_topologies = None
        if self.landmark_num_archs > 0:
            self.reset_landmark_topologies()
        if self.args.dynamic_conv_method == 'expand':
            self.args.nasbench101_template_specs = self.topologies_template_channels

    ## This belongs to interaction for now, should be removed later.
    @property
    def topologies(self):
        return self.nasbench_model_specs

    @property
    def topologies_template_channels(self):
        # this is to return the channel template. 
        if self._template_topologies is None:
            existing_keys = []
            template_spec = []
            for mid, spec in enumerate(self.topologies):
                k = tuple(spec.matrix[:,-1])
                if k not in existing_keys:
                    existing_keys.append(k)
                    template_spec.append(spec)
            self._template_topologies = template_spec
        return self._template_topologies

    @property
    def hashs(self):
        return self.nasbench_hashs

    @property
    def num_architectures(self):
        return len(self.nasbench_hashs)

    # for sparse kendall tau
    def query_gt_perfs(self, model_ids):
        """
        return the testing accuracy.
        :param model_ids: ids for the given model
        :return: gt performance of this.
        """
        return [self.nasbench.perf_rank[i][1] for i in model_ids]

    """ Aux function wrapper. """

    @staticmethod
    def change_model_spec(model, spec):
        return change_model_spec(model, spec)

    @staticmethod
    def module_forward_fn(model, input, target, criterion):
        return nasbench_model_forward(model, input, target, criterion)

    def mutate_topology(self, old_spec, mutation_rate=1.0):
        num_vertices = self.args.num_intermediate_nodes + 2
        op_spots = num_vertices - 2

        while True:
            new_matrix = deepcopy(old_spec.original_matrix)
            new_ops = list(deepcopy(old_spec.original_ops))

            # In expectation, V edges flipped (note that most end up being pruned).
            edge_mutation_prob = mutation_rate / num_vertices
            for src in range(0, num_vertices - 1):
                for dst in range(src + 1, num_vertices):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            # In expectation, one op is resampled.
            op_mutation_prob = mutation_rate / op_spots
            for ind in range(1, num_vertices - 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in self.available_ops if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            # IPython.embed()
            try:
                new_spec = ModelSpec_v2(new_matrix, new_ops)
            except Exception:
                # IPython.embed()
                continue
            if self.nasbench.is_valid(new_spec):
                _id = self.hashs.index(new_spec.hash_spec())
                return _id, new_spec
    
    def random_topology_random_nas(self):
        while True:
            new_spec = obtain_random_spec(self.args.num_intermediate_nodes + 2)
            if self.nasbench.is_valid(new_spec):
                _id = self.hashs.index(new_spec.hash_spec())
                return _id, new_spec
    
    def random_sampler(self, model, architect, args):
        """
        random sampler and update the model scenario
        :param model:
        :param architect:
        :param args:
        :return:
        """
        # IPython.embed(header='random sampler')
        if args.path_sample_method == 'random_nas':
            rand_spec_id, rand_spec = self.random_topology_random_nas()
        else: # defualt.
            rand_spec_id, rand_spec = self.random_topology()
        
        self.model_spec_id = rand_spec_id
        self.model_spec = rand_spec
        model = change_model_spec(model, rand_spec)
        # this is saved per sample.
        # self.trained_model_spec_ids.append(rand_spec_id)
        return model

    def op_sampler(self, model, architect, args):
        """
        Sample operation from model, used mainly for FairNAS procedure.
        :param model:
        :param architect:
        :param args:
        :return:
        """
        spec = self.model_spec
        ops = spec.ops
        avail_ops = self.available_ops
        try:
            op_vs_choice = np.tile(np.arange(len(avail_ops)), (len(ops)-2, 1))
            op_vs_choice = np.apply_along_axis(np.random.permutation, 1, op_vs_choice).transpose()
            for i in range(len(avail_ops)):
                new_ops = [avail_ops[ind] for ind in op_vs_choice[i]]
                spec.ops = ['input',] + new_ops + ['output']
                yield change_model_spec(model, spec)
        except ValueError as e:
            logging.warning(f'Op sampler: received exception {e}, return the original model without any op sampling.')
            yield model

    def check_valid(self, model_spec):
        try:
            return self.dataset._check_spec(model_spec)
        except:
            return False

    def serialize_model_spec(self, model_spec):
        return model_spec.hash_spec()

    def deserialize_model_spec(self, spec_str):
        try:
            idx = self.hashs.index(spec_str)
            return self.topologies[idx]
        except:
            logging.warning('Warning! Wrong spec str, out of nasbench 101 search space.')
            return None
    
    
class NasBenchSearchSpaceFixChannels(NASbenchSearchSpace):

    def __init__(self, args, full_dataset=False):
        super().__init__(args, full_dataset=full_dataset)
        self.original_model_specs = self.nasbench_model_specs
        self.original_hashs = self.nasbench_hashs
        self.sample_step_for_evaluation = 1
        self.process_nasbench_fix_channel()
        self.initialize_evaluate_pool()
        self.reset_landmark_topologies()
        # IPython.embed()

    def process_nasbench_fix_channel(self):

        Spec = namedtuple('Spec', ['model_id', 'model_spec'])
        model_specs_dict = {}

        # # key is the list version of last column.
        # for mid, spec in enumerate(self.topologies):
        #     k = tuple(spec.matrix[:,-1])
        #     if k not in model_specs_dict.keys():
        #         model_specs_dict[k] = []
        #     model_specs_dict[k].append(Spec(mid, spec))

        # for k in model_specs_dict.keys():
        #     print(k, len(model_specs_dict[k]))
        
        # overrite the current configs.
        hashs = self.nasbench_hashs
        specs = self.nasbench_model_specs
        self.nasbench_hashs = []
        self.nasbench_model_specs = []

        logging.info('Processing NASBench Fix Channel spaces: ')
        if self.args.nasbench101_fix_last_channel_config == 0:
            """ Test config 0: should return all sum 2 models """
            target_sum = self.args.nasbench101_fix_last_channel_sum
            logging.info(f'Config 0: Sum of final connection is {target_sum}')
            logging.info(f'Processing starts ...')

            for mid, spec in enumerate(specs):
                if sum(spec.matrix[1:-1,-1]) == target_sum:
                    self.nasbench_hashs.append(hashs[mid])
                    self.nasbench_model_specs.append(spec)
            
            if self.args.debug:
                logging.debug('Debugging mode: check the channel numbers')
                target_c = None 
                for mid, m in enumerate(self.topologies):
                    c = compute_vertex_channels(128, 256, m.matrix)
                    if target_c is None:
                        target_c = c 
                    assert c[1] == target_c[1], f'archi {mid}: Wrong channel number, got {c} expect {target_c} '
            logging.info(f'Finished processing.')        
            logging.info(f'Final number of architectures: {len(self.nasbench_model_specs)}')

    def random_topology_random_nas(self):
        # this is a tmp version. not sure if this differes much.
        while True:
            new_spec = obtain_random_spec(self.args.num_intermediate_nodes + 2)
            h = new_spec.hash_spec()
            if h in self.hashs:
                _id = self.hashs.index(h)
                return _id, new_spec
    
    def mutate_topology(self, old_spec, mutation_rate=1.0):
        counter = 0
        while counter < 100:
            try:
                _id, new_spec = super().mutate_topology(old_spec, mutation_rate=mutation_rate)
                return _id, new_spec
            except IndexError as e:
                counter += 1
                continue
        logging.warning('Mutation failed after 100 times. return the original arch.')
        return self.hashs.index(old_spec.hash_spec()), old_spec


class NasBenchSearchSpaceLinear(NASbenchSearchSpace):

    def __init__(self, args):
        super(NasBenchSearchSpaceLinear, self).__init__(args)
        # process only linear labels
        self.original_model_specs = self.nasbench_model_specs
        self.original_hashs = self.nasbench_hashs
        self.sample_step_for_evaluation = 1
        self.process_nasbench_linear()

    def process_nasbench_linear(self):
        """ Process nasbench linear search space. This is a much simpler search space. """
        # only take the linear architectures. a much simpler space.
        full_spec = obtain_full_model_spec(self.args.num_intermediate_nodes)
        matrix = np.eye(self.args.num_intermediate_nodes + 2, self.args.num_intermediate_nodes + 2, 1).astype(np.int)
        # indeed, we need also add ranks.
        self.nasbench_hashs = []
        self.nasbench_model_specs = []
        specs = OrderedDict()
        hashs = OrderedDict()
        for labeling in itertools.product(*[range(len(self.nasbench.available_ops))
                                            for _ in range(self.args.num_intermediate_nodes)]):
            ops = ['input', ] + [self.nasbench.available_ops[i] for i in labeling] + ['output',]
            new_spec = ModelSpec_v2(matrix.copy(), copy(ops))
            new_hash = new_spec.hash_spec()
            _id = self.original_hashs.index(new_hash)
            specs[_id] = new_spec
            hashs[_id] = new_hash

        rank_key = sorted(hashs.keys())
        self.nasbench_hashs = [hashs[_id] for _id in rank_key]
        self.nasbench_model_specs = [specs[_id] for _id in rank_key]
        self.sample_step_for_evaluation = 1
        self.initialize_evaluate_pool()
        # IPython.embed(header='check this is correct or not')
        logging.info("Linear space, totoal architecture number is {}".format(self.num_architectures))

    def evaluate_model_spec_id_pool(self):
        return self.evaluate_model_spec_ids


class NasBenchSearchSpaceSubsample(NASbenchSearchSpace):

    # keep track of original space ids, because new id will be flushed.
    rank_id_in_original_nasbench = []

    def __init__(self, args):
        super(NasBenchSearchSpaceSubsample, self).__init__(args)
        self.original_model_specs = self.nasbench_model_specs
        self.original_hashs = self.nasbench_hashs
        self.sample_step_for_evaluation = 1
        self.process_subsample_space()

    def process_subsample_space(self):
        # raise NotImplementedError('finish later')
        if self.args.num_archs_subspace > 0:
            sample_num = min(self.args.num_archs_subspace, self.num_architectures)                        
            subspace_ids = sorted([int(a) for a in np.random.choice(
                len(self.nasbench_model_specs), sample_num, replace=False)])
            
        else:
            sample_num = self.num_architectures
            subspace_ids = range(sample_num)
            
        self.rank_id_in_original_nasbench = subspace_ids
        self.nasbench_hashs = [self.original_hashs[_id] for _id in subspace_ids]
        self.nasbench_model_specs = [self.original_model_specs[_id] for _id in subspace_ids]
        self.initialize_evaluate_pool()
        logging.info("Random subspace with {} architectures: {}".format(self.num_architectures, subspace_ids[:100]))
        logging.info("Evaluation architecture pool: {}".format(self.evaluate_model_spec_ids))


def nodes_to_key(nodes):
    # always [0, 1, 2, ..., num_intermediate_nodes]
    # nodes = range(len(nodes))
    # return ','.join(map(str, nodes))
    return len(nodes)

def key_to_nodes(key):
    # return [int(a) for a in key.split(',')]
    return list(range(key))

def model_spec_to_involving_nodes(spec):
    matrix = spec.matrix.copy()
    active_nodes = np.argwhere(matrix.sum(axis=1)[1:-1] > 0).reshape(-1)
    return active_nodes.tolist(), matrix


def permunate_ops_all(n, OPS):
    if n == 0:
        yield []
    elif n == 1:
        for o in OPS:
            yield [o,]
    else:
        for o in OPS:
            for rest_ops in permunate_ops_all(n-1, OPS):
                yield [o,] + rest_ops


def permunate_ops_last_node(n, OPS, default_pos=0):
    for o in OPS:
        yield [OPS[default_pos], ] * (n-1) + [o,]


def permutate_ops_given_topology(matrix, OPS, permutate_last=True):
    # print('permutate under topology matrix ', matrix) 
    node = matrix.shape[0] - 2
    if permutate_last:
        all_ops = permunate_ops_last_node(node, OPS, default_pos=0)
    else:
        all_ops = permunate_ops_all(node, OPS)
    for ops in all_ops:
        ops = ['input', ] + ops + ['output',]
        copy_matrix = matrix.copy()
        a = ModelSpec_v2(copy_matrix, ops)
        if a.valid_spec:
            yield a


class NasBenchSearchSpaceFairNasTopology(NASbenchSearchSpace):

    def __init__(self, args):
        super(NasBenchSearchSpaceFairNasTopology, self).__init__(args)
        self.nasbench_involving_nodes = OrderedDict()
        for ind, spec in enumerate(self.topologies):
            active_nodes, matrix = model_spec_to_involving_nodes(spec)
            key = nodes_to_key(active_nodes)
            if key in self.nasbench_involving_nodes.keys():
                self.nasbench_involving_nodes[key].append(ind)
            else:
                self.nasbench_involving_nodes[key] = [ind, ]
        self.nasbench_topo_sample_probs = []
        for k, v in self.nasbench_involving_nodes.items():
            logging.debug(f'involving nodes {k} : num arch {len(v)}')
            self.nasbench_topo_sample_probs.append(len(v))
        self.nasbench_topo_sample_probs = list(reversed(self.nasbench_topo_sample_probs))

    def nasbench_sample_matrix_from_list(self, nodes, probs):
        """
        Recursively sample from the list of data by a prob.
        This cooperates with fairnas topology sampler.
        Fair sampling.

        :param nodes: [1, ... interm,] node id as a list
        :param probs: probability to sample an list with length equal to probs, len(probs) == len(data)
        :return:
        """

        def normalize(probs):
            return list(map(lambda x: float(x / sum(probs)), probs))

        if len(nodes) == 0:
            return [None,]
        else:
            try:
                total = self.args.num_intermediate_nodes
                probs = normalize(probs)
                num_sample = np.random.choice(np.arange(len(nodes) + 1), 1, p=probs)
                sample_nodes = sorted(np.random.choice(nodes, num_sample, replace=False))
                rest_nodes = list(set(nodes) - set(sample_nodes))
                new_probs = probs[:len(rest_nodes) + 1]
                # nasbench matrix including input and output.
                topo_matrices_ids = self.nasbench_involving_nodes[nodes_to_key(sample_nodes)]
                sample_id = np.random.choice(topo_matrices_ids, 1)[0]
                sample_matrix = self.topologies[sample_id].matrix.copy()
                if sample_matrix.shape[0] == total + 2:
                    # terminate the looping.
                    return [sample_matrix, None]
                else:
                    # Make sample nodes to full matrix spec.
                    sample_nodes = [0,] + sample_nodes + [total + 1]
                    # make new_matrix[sample_nodes,:][:, sample_nodes] = sample_matrix
                    matrix = np.zeros([total + 2, total + 2], dtype=int)
                    _matrix = matrix[sample_nodes,:]
                    _matrix[:, sample_nodes] = sample_matrix
                    matrix[sample_nodes,:] = _matrix
                    return [matrix,] + self.nasbench_sample_matrix_from_list(rest_nodes, new_probs)
            except Exception as e:
                logging.error(f'{e}')
                IPython.embed(header='Check mistake of nasbench_sample_matrix_from_list')


class NasBenchSearchSpaceICLRInfluenceWS(NasBenchSearchSpaceSubsample):
    arch_hash_by_group = {}
    arch_ids_by_group = {}

    # preprocess this search space.
    def process_subsample_space(self):
        # composing the linear search space.
        # Variates of ops, but topology is sampled from a pool.
        nodes = self.args.num_intermediate_nodes
        AVAILABLE_OPS = self.nasbench.available_ops
        logging.info("Processing NASBench WS influence Search Space ...")
        permutate_op_fn = partial(permutate_ops_given_topology,
                                  permutate_last=not self.args.nasbench_search_space_ws_influence_full)
        logging.info("Permutating the last node only? {}".format(
            not self.args.nasbench_search_space_ws_influence_full))

        subspace_ids = []
        subspace_model_specs_dict = {}
        # make all possible matrix:
        for i in range(nodes):
            matrix = np.zeros((nodes + 2, nodes + 2), dtype=np.int)
            matrix[nodes, -1] = 1  # connect output to node n-1.
            matrix[i, -2] = 1  # connect last node to one of the previous node.
            if i > 0:
                if i > 1:
                    matrix[0:i, 1:i + 1] = np.triu(np.ones((i, i), dtype=np.int))
                else:
                    matrix[0, 1] = 1

            logging.info(f'Node {i}-{nodes} connection: {matrix}')

            self.arch_hash_by_group[i] = []
            self.arch_ids_by_group[i] = []

            for spec in permutate_op_fn(matrix, AVAILABLE_OPS):
                hash = spec.hash_spec()
                spec.resume_original()
                try:
                    _id = self.nasbench_hashs.index(hash)
                except ValueError as e:
                    logging.error("Spec is not valid here: {}".format(e))
                    logging.error(spec)
                    continue
                # subspace_ids.append(_id)
                if hash not in subspace_model_specs_dict.keys():
                    # only keep one spec.
                    subspace_model_specs_dict[hash] = spec
                    self.arch_hash_by_group[i].append(hash)
                    self.arch_ids_by_group[i].append(_id)
                    subspace_ids.append(_id)

        # count = 0
        # for i in range(nodes):
        #     n_g = []
        #     n_id = []
        #     # removing the duplicated items
        #     logging.info(f"Process rank group {i}, original length {len(self.arch_ids_by_group[i])} ... ")
        #     for _id, h in zip(self.arch_ids_by_group[i], self.arch_hash_by_group):
        #         if _id not in n_id:
        #             n_id.append(_id)
        #             n_g.append(h)
        #     self.arch_ids_by_group[i] = n_id
        #     self.arch_hash_by_group[i] = n_g
        #     assert len(n_id) == len(n_g)
        #     count += len(n_id)
        #     logging.info("Length after processing: {}".format(self.arch_ids_by_group[i]))

        sort_ids = np.argsort(subspace_ids)
        sort_subspace_ids = [subspace_ids[i] for i in sort_ids]
        self.nasbench_model_specs_prune = [self.original_model_specs[i] for i in sort_subspace_ids]
        self.nasbench_hashs = [self.original_hashs[_id] for _id in sort_subspace_ids]
        self.nasbench_model_specs = [subspace_model_specs_dict[h] for h in self.nasbench_hashs]
        self.initialize_evaluate_pool()
        logging.info("Totally {} architectures: {}".format(self.num_architectures, subspace_ids[:100]))
        logging.info("Evaluation architecture pool: {}".format(self.evaluate_model_spec_ids))
