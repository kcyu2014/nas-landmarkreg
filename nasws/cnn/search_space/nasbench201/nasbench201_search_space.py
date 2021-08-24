#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/26 下午4:16
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================
import IPython
import logging
import numpy as np
from copy import deepcopy
import random

from numpy.lib.arraysetops import isin
from ..api import CNNSearchSpace
from .model import NasBench201Net
from .nasbench201_dataset import NASBench201Benchmark
from .lib.models import CellStructure
from .model_search import build_nasbench201_search_model


def check_valid_node_num(spec):
    """Check the valid num of node for each NASBench 201 dataset.

    
    Parameters
    ----------
    spec : Structure
    Returns
    -------
    num of valid node.
    """
    nodes = {0: True}
    for i, node_info in enumerate(spec.nodes):
        sums = []
        for op, xin in node_info:
            if op == 'none' or nodes[xin] == False: x = False
            else: x = True
            sums.append(x)
        nodes[i+1] = sum(sums) > 0
    return sum([1 if n else 0 for n in nodes.values()])


def random_architecture_func(max_nodes, op_names):
    # It seems that the architecture connection in NASBench-201 is always 001012 ! yes! so no need actually ...
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
    """Computes the architecture for a child of the given parent architecture.
    The parent architecture is cloned and mutated to produce the child architecture. The child architecture is mutated by randomly switch one operation to another.
    """
    def mutate_arch_func(parent_arch):
        child_arch = deepcopy(parent_arch)
        node_id = random.randint(0, len(child_arch.nodes)-1)
        node_info = list(child_arch.nodes[node_id])
        snode_id = random.randint(0, len(node_info)-1)
        xop = random.choice(op_names)
        while xop == node_info[snode_id][0]:
            xop = random.choice(op_names)
        node_info[snode_id] = (xop, node_info[snode_id][1])
        child_arch.nodes[node_id] = tuple(node_info)
        return child_arch
    return mutate_arch_func


class NASBench201SearchSpace(CNNSearchSpace):
    """NASBench102 search space wrapper. 

    """
    
    # override some meta-variables
    sample_step_for_evaluation = 1
    top_K_complete_evaluate = 200
    landmark_num_archs = 10
    # stores the pool of evaluate ids. Fixed after initialized.
    evaluate_ids = None
    # Current pool of ids, can change from time to time.
    evaluate_model_spec_ids = None

    def __init__(self, args, full_dataset=False):
        super(NASBench201SearchSpace, self).__init__(args)
        self.topology_fn = NasBench201Net
        self.sample_step_for_evaluation = 1 if not args.debug else 30
        dataset = args.nasbench102_dataset
        self.dataset = NASBench201Benchmark(args.data + f'/nasbench102/nasbench102-v1-{dataset}.json',
                                            dataset, full_dataset)
        self.model_fn = build_nasbench201_search_model
        self._construct_search_space()
        self.initialize_evaluate_pool()
        # self.initialize_search_space()
        self.num_classes = {'cifar10': 10, 'cifar100': 100, 'imagenet':1000}[dataset]

    @staticmethod
    def change_model_spec(model, spec):
        """ change spec accordingly """
        if hasattr(model, 'module'):
            model.module.arch_cache = spec
        else:
            model.arch_cache = spec
        return model

    @staticmethod
    def module_forward_fn(model, input, target, criterion):
        """ forward function to compute in this search space. """
        logits, _ = model(input)
        loss = criterion(logits, target)
        return loss, logits, None

    def mutate_topology(self, old_spec, mutation_rate=1.0):
        """ Round the mutation rate to int and apply this. """
        # check this is correct.
        f = mutate_arch_func(self.dataset.available_ops)
        new_spec = old_spec
        for i in range(int(mutation_rate)):
            new_spec = f(new_spec)
        # self.dataset
        new_hash = self.dataset._hash_spec(new_spec)
        try:
            return self._hashs.index(new_hash), new_spec
        except:
            logging.warning("wrong hash spec for new ")
            return 0, new_spec

    def random_topology_random_nas(self):
        """Generate random NAS cell.
        
        Returns
        -------
        [type]
            [description]
        """
        new_spec = random_architecture_func(self.args.num_intermediate_nodes, self.dataset.available_ops)()
        new_hash = self.dataset._hash_spec(new_spec)
        try:
            return self._hashs.index(new_hash), new_spec
        except IndexError as e:
            IPython.embed()
            raise e

    def op_sampler(self, model, architect, args):
        # for fair-nas sampling.
        
        # Topology is fixed, so the node is always like 
        # (op, 0), [(op,0), (op, 1)] ... up to [(op, 0), ... (op, n-1)] for n nodes.
        # because the topology is not changed always, so we could just print these out.
        def create_op_choices(_avail_ops, num):
            op_vs_choice = np.tile(np.arange(len(_avail_ops)), (num, 1))
            op_vs_choice = np.apply_along_axis(np.random.permutation, 1, op_vs_choice).transpose()
            return op_vs_choice
        
        def create_cell_structure(op_names):
            # create one architecture.
            genotypes = []
            _id = 0
            for i in range(1, args.num_intermediate_nodes):
                xlist = []
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    xlist.append((op_names[_id], j))
                    _id += 1
                genotypes.append(tuple(xlist))
            return CellStructure(genotypes)
        
        max_node = args.num_intermediate_nodes
        avail_ops = self.dataset.available_ops

        try:
            op_choices = create_op_choices(avail_ops, int(max_node * (max_node - 1) / 2))
            for i in range(len(avail_ops)):
                # each ops we generate one
                new_spec = create_cell_structure([avail_ops[_id] for _id in op_choices[i]])
                yield self.change_model_spec(model, new_spec)
        except ValueError as e:
            logging.warning(f"Op sampler: receive exception {e}, return the original model without any sampling")
            yield model

    def random_sampler(self, model, architect, args):
        if args.path_sample_method == 'default':
            model_spec_id, model_spec = self.random_topology()
        elif args.path_sample_method == 'random_nas':
            model_spec_id, model_spec = self.random_topology_random_nas()
        else:
            raise ValueError("Path sampling not supported, {}".format(args.path_sample_method) )

        self.model_spec_id = model_spec_id
        self.model_spec = model_spec

        new_model = self.change_model_spec(model, model_spec)
        # self.trained_model_spec_ids.append(model_spec_id)
        return new_model
    
    def serialize_model_spec(self, model_spec):
        if isinstance(model_spec, CellStructure):
            return model_spec.hash_spec()
        elif isinstance(model_spec, str):
            return model_spec
        elif isinstance(model_spec, list):
            return CellStructure(model_spec).hash_spec()
        else:
            raise ValueError('Model spec should be either a str or a CellStructure')

    def deserialize_model_spec(self, spec_str):
        return CellStructure.str2structure(spec_str)
    
    def check_valid(self, model_spec):
        return model_spec.check_valid()

    # def _construct_search_space(self):
    #     """ constructing search space based on dataset. """
    #     _hashs = []
    #     _model_specs = []
    #     _perfs = []
    #     _losses = []
        
    #     # load the dataset
    #     for h, s in self.dataset.hash_dict.items():
    #         if check_valid_node_num(s) > self.args.num_intermediate_nodes:
    #             continue
    #         if not s.check_valid() and self.args.nasbench102_use_valid_only:
    #             continue
    #         if self.args.nasbench102_use_isomorphic:
    #             h = s.to_unique_str(self.args.nasbench102_use_isomorphic_consider_zeros)
    #             if h in _hashs:
    #                 continue
    #         _hashs.append(h)
    #         _model_specs.append(s)
    #         _perfs.append(self.dataset.query_perf(s))
    #         _losses.append(self.dataset.query_loss(s))

    #     sorted_index = np.argsort(_perfs).tolist()
    #     # here we update the search space subsample
    #     if self.args.search_space == 'nasbench102_subsample':
    #         logging.info(f'NASBench102 using subsample space {self.args.num_archs_subspace}')
    #         # reduce the search space to a small number
    #         import random
    #         IPython.embed(header='check the sorted index sampling.')
    #         sorted_index = random.sample(sorted_index, k=self.args.num_archs_subspace)

    #     self._hashs = [_hashs[i] for i in sorted_index]
    #     self._model_specs = [_model_specs[i] for i in sorted_index]
    #     self._perfs = [_perfs[i] for i in sorted_index]
    #     self._losses = [_losses[i] for i in sorted_index]
    #     self._ranks = list(range(len(self._hashs)))

    #     logging.info(f"{self.__class__.__name__} loaded. total trained model points {len(self._hashs)}")
    #     self.landmark_num_archs = self.args.landmark_num_archs
    #     self.reset_landmark_topologies()
    #     self.initialize_evaluate_pool()