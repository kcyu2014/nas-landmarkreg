### Darts search space is the most commonly used one. And we have the ground-truth results here.


# current goal is to make a search space, that has maximum amount of code-sharing
# with NASbench / and in future ImageNet space.
import logging
import os
import random
from collections import deque
from copy import copy

import IPython
import numpy as np

from nasws.cnn.policy.darts_policy.genotypes import Genotype
from nasws.cnn.search_space.search_space import CNNSearchSpace
from .dartsbench import DartsModelSpec, DARTSBench, random_darts_genotype
from .utils import change_model_spec
from .model import NetworkCIFAR, NetworkImageNet
from .model_search import DartsNetworkCIFARSearch
from .model_search_imagenet import DartsNetworkImageNetSearch
from .operations import PRIMITIVES


class DartsSearchSpace(CNNSearchSpace):
    
    available_ops = PRIMITIVES
    num_classes = 10
    new_hashs = []
    new_specs = []

    def __repr__(self):
        return 'DartsSearchSpace: total archs {}'.format(self.num_architectures)

    def __init__(self, args, full_dataset=False):
        super(DartsSearchSpace, self).__init__(args)
        self.topology_fn = NetworkCIFAR if 'cifar' in args.dataset else NetworkImageNet
        self.model_fn = DartsNetworkCIFARSearch if 'cifar' in args.dataset else DartsNetworkImageNetSearch
        # process those trained architecture in this space.
        self.dataset = DARTSBench(os.path.join(self.args.data, 'dartsbench/dartsbench_v1.json'))
        self._construct_search_space()

    def initialize_evaluate_pool(self):
        self.evaluate_ids = [i for i in range(self.num_architectures)]

        # remove the landmark id from eval_ids
        if self.landmark_num_archs > 0:
            self.evaluate_ids = list(set(self.evaluate_ids) - set(self._landmark_ids))

        self.evaluate_model_spec_ids = deque(self.evaluate_ids)
        if len(self.evaluate_model_spec_ids) > self.top_K_complete_evaluate:
            self.evaluate_model_spec_ids = deque(sorted(
                np.random.choice(self.evaluate_model_spec_ids, self.top_K_complete_evaluate, replace=False).tolist()))

        logging.info(f"Architecture IDS to evaluate {self.evaluate_model_spec_ids}")

    def _potential_register_model_spec(self, model_spec):
        """ return id and spec, potentially register if not seen. 
        This logic is fundamentally flawed. you should never add it to the model pool 
        because the kdt computation will be a huge mess!!!

        """
        hash = model_spec.hash_spec()
        if hash in self.hashs:
            return self.hashs.index(hash), model_spec
        else:
            logging.debug('sample a new model here! ')
            self.new_hashs.append(hash)
            self.new_specs.append(model_spec)
            return None, model_spec

    def random_topology_random_nas(self):
        """ return id, model_spec """
        rand_geno = random_darts_genotype(self.args.num_intermediate_nodes)
        model_spec = DartsModelSpec.from_darts_genotype(rand_geno)
        return self._potential_register_model_spec(model_spec)

    def random_topology(self):
        """ Random spec """
        rand_spec_id = self.random_ids(1)[0]
        rand_spec = self.topologies[rand_spec_id]
        return rand_spec_id, rand_spec

    @staticmethod
    def change_model_spec(model, spec):
        return change_model_spec(model, spec)

    @staticmethod
    def module_forward_fn(model, input, target, criterion):
        """ forward function to compute in this search space. """
        logits, aux_logits = model(input)
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        return loss, logits, aux_logits

    def mutate_topology(self, old_spec, mutation_rate=1.0):
        """ mutate topology return a model spec. """
        old_geno = old_spec.to_darts_genotype()
        normal_cell = old_geno.normal
        reduce_cell = old_geno.reduce
        # keep concat the same.
        def mutate_cell(cell):
            cell = [list(c) for c in cell]
            print("original cell ", cell)
            num_nodes = int(len(cell) / 2)
            num_edges = int((2 + num_nodes + 1) * num_nodes / 2)
            edge_mutation_prob = mutation_rate / (num_nodes * 2)
            op_mutation_prob = mutation_rate / len(PRIMITIVES)
            # mutate connections
            for i in range(num_nodes * 2):
                curr_idx = i // 2
                if random.random() < edge_mutation_prob:
                    choices = list(set(range(0, curr_idx + 2)) - {cell[i][0]})
                    cell[i][1] = random.choice(choices)
                if random.random() < op_mutation_prob:
                    choices = list(set(PRIMITIVES) - {cell[i][1]})
                    cell[i][0] = random.choice(choices)
            print('mutated cell, ', cell)
            return cell

        new_model_spec = DartsModelSpec.from_darts_genotype(
            Genotype(mutate_cell(normal_cell), copy(old_geno.normal_concat),
                     mutate_cell(reduce_cell), copy(old_geno.reduce_concat))
        )
        return self._potential_register_model_spec(new_model_spec)

    def op_sampler(self, model, architect, args):
        """ sampling operation """
        # key is to assign model_spec
        def create_op_choices(_avail_ops, num):
            op_vs_choice = np.tile(np.arange(len(_avail_ops)), (num, 1))
            op_vs_choice = np.apply_along_axis(np.random.permutation, 1, op_vs_choice).transpose()
            return op_vs_choice

        spec = self.model_spec
        g = spec.to_darts_genotype()
        num_nodes = len(g.normal), len(g.reduce)
        avail_ops = self.available_ops
        try:
            normal_choices = create_op_choices(avail_ops, num_nodes[0])
            reduce_choices = create_op_choices(avail_ops, num_nodes[1])
            for i in range(len(avail_ops)):
                normal_c = [ (avail_ops[j], g.normal[indx][1])  for indx, j in enumerate(normal_choices[i]) ]
                reduce_c = [ (avail_ops[j], g.reduce[indx][1])  for indx, j in enumerate(reduce_choices[i]) ]
                new_spec = DartsModelSpec.from_darts_genotype(Genotype(normal_c, g.normal_concat, reduce_c, g.reduce_concat))
                yield self.change_model_spec(model, new_spec)
        except ValueError as e:
            logging.warning(f"Op sampler: receive exception {e}, return the original model without any sampling")
            yield model
        
    def random_sampler(self, model, architect, args):
        """ random topology """
        if args.path_sample_method == 'default':
            model_spec_id, model_spec = self.random_topology()
        else:
            raise ValueError("Path sampling not suported for DARTS-NDS, {}".format(args.path_sample_method))

        # used in other class
        self.model_spec_id = model_spec_id
        self.model_spec = model_spec
        # keep sampling this one. if random sampler is called, this model-spec is updated.
        new_model = self.change_model_spec(model, model_spec)
        # self.trained_model_spec_ids.append(model_spec_id)
        return new_model
    
    def check_valid(self, model_spec):
        if isinstance(model_spec, DartsModelSpec):
            return model_spec.check_valid()
        elif isinstance(model_spec, (tuple, list)):
            return DartsModelSpec.from_darts_genotype(model_spec).check_valid()
        else:
            logging.info('Wrong spec format, return False')
            return False

# class ImagenetDartsSearchSpace(DartsSearchSpace):

#     num_classes = 1000

#     def __init__(self, args, full_dataset=False):
#         super(DartsSearchSpace, self).__init__(args)
#         assert args.dataset == 'imagenet', 'wrong dataset in Imagenet Darts Search Space'
#         self.topology_fn = NetworkImageNet
#         self.model_fn = DartsNetworkImageNetSearch
#         self.dataset = DARTSBench(os.path.join(self.args.data, 'nni_nds'), dataset='imagenet', proposer='darts')
#         self._construct_search_space()
 