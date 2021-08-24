### Follow nasbench, create a dataset, even only with a few data points.
import json
import logging
import os
import random

import IPython
import torch
import time
import numpy as np

import utils
from ..api import BenchmarkDatasetTemplate, _NumpyEncoder
from nasws.cnn.search_space.search_space import OpSpecTemplate, CellSpecTemplate
from nasws.cnn.policy.darts_policy.genotypes import PRIMITIVES, Genotype, \
    transfer_NAO_arch_to_genotype, transfer_ENAS_arch_to_genotype


ENAS_PATH = '/home/yukaiche/pycharm/automl/experiments/iclr-resubmission/ENAS-train-from-scratch'
NAO_PATH = '/home/yukaiche/pycharm/automl/experiments/iclr-resubmission/NAO-train-from-scratch'
DARTS_PATH = '/home/yukaiche/pycharm/automl/experiments/iclr-resubmission/DARTS-train-from-scratch'
RANDOM_PATH = '/home/yukaiche/pycharm/automl/experiments/iclr-resubmission/Guided-RANDOM-train-from-scratch'
CVPR_PATH='/home/yukaiche/pycharm/automl/experiments/iclr-resubmission/RANKLOSS-train-from-scratch'

P_NORMAL = [0, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2]
P_REDUCE = [0, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1]


def random_cell(node_num, num_sample=2, OPS=PRIMITIVES, weights=None):
    concat = list(range(2, node_num + 2))
    cell = []
    for step in range(node_num):

        prev_ids = np.random.choice(
            np.arange(0, 2 + step, dtype=np.int), num_sample,
            replace=True)

        for _id in prev_ids:
            if weights is not None:
                weights = np.array(weights)

            else:
                weights = np.ones(len(OPS))
            weights /= weights.sum()
            op_id = np.random.choice(np.arange(0, len(OPS), dtype=np.int),
                                     1, p=weights)[0]
            cell.append((OPS[op_id], _id))
    return cell, concat


def random_darts_genotype(intermediate_node):
    normal_cell, normal_concat = random_cell(intermediate_node)
    reduce_cell, reduce_concat = random_cell(intermediate_node)
    return Genotype(normal_cell, normal_concat, reduce_cell, reduce_concat)


def guided_random_darts_genotype(intermediate_node):
    normal_cell, normal_concat = random_cell(intermediate_node, weights=P_NORMAL)
    reduce_cell, reduce_concat = random_cell(intermediate_node, weights=P_REDUCE)
    return Genotype(normal_cell, normal_concat, reduce_cell, reduce_concat)


class DartsOpSpec(OpSpecTemplate):
    typename = 'DartsSearchSpaceOpToken'
    op_choices = {
        'op1_name': PRIMITIVES,
        'op2_name': PRIMITIVES
    }
    _type = 'list'
    auxilary_name = {
        -1: 'input',
        -2: 'prev_input',
        -3: 'output'
    }
    available_ops = PRIMITIVES
    available_ops_all = PRIMITIVES + list(auxilary_name.values())

    @classmethod
    def build_op_from_list(cls, input_list):
        """
        Make sure the list constructed is in this format
        [prev_input, input, [op1, op2], ... output]
        :param input_list:
        :return:
        """
        # process and checking the data.
        np_data = np.asanyarray(input_list, dtype=object)
        if any([i in cls.auxilary_name.values() for i in input_list]):
            return cls(input_list, 'list')

        if np.ndim(np_data) == 1:
            _list = [(input_list[i], input_list[i+1]) for i in range(0,len(input_list),2)]
            np_data = np.asanyarray(_list, dtype=object)

        if not all([i in cls.available_ops_all for i in np_data.reshape(-1).tolist()]):
            raise ValueError("List to construct Darts ops is wrong. \n"
                             "recieve {} \n"
                             "expect {}.".format(input_list, cls.available_ops_all))

        return cls(input_list, 'list')

    @property
    def labeling(self):
        def _op_encoding(op_list):
            return len(PRIMITIVES) * PRIMITIVES.index(op_list[0]) + PRIMITIVES.index(op_list[1])

        # because the len(label) == len(in_edge) == len(out_edge), lets group PRIMITIVE op pair in one.
        return [-2, -1] + [_op_encoding(ops) for ops in self.ops[2:-1]] + [-3]


class DartsCellSpec(CellSpecTemplate):

    def __init__(self, matrix, ops, **kwargs):
        super(DartsCellSpec, self).__init__(matrix, DartsOpSpec.build_op_from_list(ops), **kwargs)
        self.check_valid()

    # def _prune(self):
    #     super(DartsCellSpec, self)._prune()

    def check_valid(self):
        """
        Checking the cell spec is correct. As in DARTS, the format is strictly in
            Genotype( [node 1, op1, node 2, op2] for node _i)

        :return:
        """
        valid = True
        if not self.ops.check_consistency(self.matrix):
            valid = False

        res = np.sum(self.matrix, axis=0)
        if not np.all(np.where(res[2:-1] == 2, True, False)):
            logging.debug(f"DartsCellSpec check valid error: not all interemdiate nodes has 2 ops. {res[2:-1]}")
            valid = False
        self.valid_spec = valid
        return valid


class DartsModelSpec(object):
    """
    Placeholder for Darts Model Spec
    """

    @property
    def valid_spec(self):
        return self.normal_spec.valid_spec and self.reduce_spec.valid_spec

    def __repr__(self):
        return str(self._darts_genotype)

    def __str__(self):
        return str(self._darts_genotype)

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __init__(self, normal_matrix, normal_ops, reduce_matrix, reduce_ops, data_format='channel_last'):
        super(DartsModelSpec, self).__init__()
        self.normal_spec = DartsCellSpec(normal_matrix, normal_ops)
        self.reduce_spec = DartsCellSpec(reduce_matrix, reduce_ops)
        self.specs = [self.normal_spec, self.reduce_spec]
        self.spec_names = ['normal', 'reduce']
        self._darts_genotype = None

    def hash_spec(self):
        return ''.join([cell.hash_spec() for cell in self.specs])

    def check_valid(self):
        return self.normal_spec.check_valid() and self.reduce_spec.check_valid()
        
    """ Create from the genotypes of each algorithms """
    @staticmethod
    def darts_genotype_concat_to_matrix_ops(cell, concat=None, compute_concat=True):
        """

        :param cell:
        :param concat:
        :param compute_concat: replace the concat. if concat is not None, ignore this.
        :return:
        """
        compute_concat = False if concat is not None else compute_concat
        ALLOWED_OPS = DartsOpSpec.available_ops
        num_intermediate_node = len(cell) // 2
        matrix = np.zeros((num_intermediate_node + 3, num_intermediate_node + 3), dtype=np.int)
        ops = []
        is_output = [True, ] * (num_intermediate_node + 2)

        for i in range(num_intermediate_node):
            _op = []
            curr_node = i + 2

            for j in range(2):
                prev_op, prev_node = cell[i*2 + j]
                _op.append(prev_op)
                is_output[prev_node] = False
                matrix[prev_node][curr_node] += 1 # may be repeat
            # append the op
            ops.append(_op)

        if not compute_concat:
            is_output = [True if ind in concat else False for ind in range(num_intermediate_node + 2)]

        for input_node, connect_to_output in enumerate(is_output):
            matrix[input_node][num_intermediate_node + 2] = 1 if connect_to_output else 0
        matrix[0][1] = 1    # set this to not prune prev_input, input
        matrix = matrix.astype(np.int)
        ops = ['prev_input', 'input'] + ops + ['output']
        return matrix, ops


    @classmethod
    def from_darts_genotype(cls, genotype: Genotype):
        n_matrix, n_ops = cls.darts_genotype_concat_to_matrix_ops(genotype.normal, genotype.normal_concat)
        r_matrix, r_ops = cls.darts_genotype_concat_to_matrix_ops(genotype.normal, genotype.normal_concat)
        a = cls(n_matrix, n_ops, r_matrix, r_ops)
        a._darts_genotype = genotype
        return a

    @classmethod
    def from_nao_genotype(cls, genotype):
        return cls.from_darts_genotype(transfer_NAO_arch_to_genotype(genotype))

    @classmethod
    def from_enas_genotype(cls, genotype):
        return cls.from_darts_genotype(transfer_ENAS_arch_to_genotype(genotype))

    """ To specific genotype for each algorithms. """

    def to_darts_genotype(self):
        if self._darts_genotype:
            return self._darts_genotype
        else:
            raise ValueError("Not yet supported.")

    def to_enas_genotype(self):
        pass

    def to_nao_genotype(self):
        pass

    # test the to json and from json.


class DARTSBench(BenchmarkDatasetTemplate):
    """
    mimic NASBench, follow the API design
    to support some basic options later.
    quickly add support for landmark architectures
    """

    fixed_statistics_keys = ['genotype', 'trainable_parameters', 'meta_information']
    computed_statistics_keys = ['final_train_accuracy', 'final_validation_accuracy','final_test_accuracy']

    def __init__(self, dataset_file, model_arch_file='dartsbench_model_specs.json', seed=None, config=None):
        self.dataset_file = dataset_file
        # self.model_arch_file = os.path.join(os.path.dirname(), model_arch_file)
        self.valid_epochs = {600}
        super(DARTSBench, self).__init__(dataset_file, seed, config)

    def create_model_spec(self, fixed_data):
        """ Create a genotype accordingly. """
        create_geno = lambda l: Genotype(l[0], l[1], l[2], l[3])
        return DartsModelSpec.from_darts_genotype(create_geno(fixed_data['genotype']))

    # def load_dataset_file(self, dataset_file):
    #     try:
    #         d = utils.load_json(dataset_file)
    #         self.fixed_statistics = d['fixed_stat']
    #         self.computed_statistics = d['compute_stat']
    #         create_geno = lambda l: Genotype(l[0], l[1], l[2], l[3])
    #         self.hash_dict = {h: DartsModelSpec.from_darts_genotype(create_geno(fix_stat['genotype']))
    #                           for h, fix_stat in self.fixed_statistics.items()}
    #     except FileNotFoundError:
    #         print("File not found, creating the dataset by loading from the disk.")
    #         self.preprocess_dataset_from_given_files()
    #         self.save_dataset_file()
    #
    # def save_dataset_file(self):
    #     save_data = {'fixed_stat': self.fixed_statistics, 'compute_stat': self.computed_statistics}
    #     utils.save_json(save_data, self.dataset_file)

    def preprocess_dataset_from_given_files(self):
        """ preprocess dataset based on previous trained results. """
        import glob
        from nasws.cnn.policy.darts_policy.genotypes import darts_final_genotype_by_seed, \
            enas_final_genotypes_by_seed, random_generate_genotype_by_seed, nao_final_genotypes_by_seed

        for ckpt, geno_fn in zip([ENAS_PATH, NAO_PATH, DARTS_PATH, RANDOM_PATH],
                                 [enas_final_genotypes_by_seed, nao_final_genotypes_by_seed,
                                  darts_final_genotype_by_seed, random_generate_genotype_by_seed]):

            print(ckpt.split('/')[-1])
            _checkpoint_pts = glob.glob(ckpt + '/*/checkpoint.pt')
            _perf_dict = {}
            name = ckpt.split('/')[-1].split('-train')[0]

            for p in _checkpoint_pts:
                seed = int(p.split('seed_')[1].split('-eval')[0])
                genotype = geno_fn(seed)
                model_spec = DartsModelSpec.from_darts_genotype(genotype)
                metadata = self.train_and_evaluate(None, None, None, p)
                metadata['meta_information'] = {'approach': name, 'path': p}
                self.update_statistics_by_metadata(model_spec, metadata)

        logging.info("Loading trained model from given paths.")

    def update_statistics_by_metadata(self, model_spec: DartsModelSpec, metadata, epoch=600):
        hash = model_spec.hash_spec()
        fix_stat_values = [model_spec.to_darts_genotype(), 0, metadata['meta_information']]
        eval_results = metadata['evaluation_results'][-1]
        compute_stat_values = [eval_results['train_accuracy'],
                               eval_results['validation_accuracy'],
                               eval_results['test_accuracy']]
        # craete this hash dict
        if hash not in self.fixed_statistics.keys():
            self.fixed_statistics[hash] = {}
            self.computed_statistics[hash] = {}

        for k, v in zip(self.fixed_statistics_keys, fix_stat_values):
            self.fixed_statistics[hash][k] = v

        try:
            num_run = len(self.computed_statistics[hash][epoch])
            self.computed_statistics[hash][epoch].append({})
        except KeyError:
            num_run = 0
            self.computed_statistics[hash][epoch] = [{}]
        except Exception as e:
            num_run = None
            print(e)
            IPython.embed()

        self.computed_statistics[hash][epoch][num_run] = dict(zip(self.computed_statistics_keys, compute_stat_values))
        self.hash_dict[hash] = model_spec

    def train_and_evaluate(self, model_spec, config, model_dir, preload_from_folder=None):
        """ support load training results from folder """
        if preload_from_folder:
            p = preload_from_folder
            mtime = os.path.getmtime(p)
            # quadro = True if 'quadro' in p.lower() else False
            model_save_path = p
            state = torch.load(model_save_path)
            init_epoch = state['epoch']
            best_acc = state['best_acc']
            # IPython.embed(header='pause here to check the init_epoch')
            logging.debug(f"{init_epoch + 1}, best accuracy {state['best_acc']}, "
                          f"last update {time.ctime(mtime)}")
            metadata = {
                'evaluation_results':[{
                    'training_time':0,
                    'train_accuracy':0,
                    'validation_accuracy':0,
                    'test_accuracy': best_acc
                }],
                'trainable_params':-1,
            }
            return metadata
        else:
            return super(DARTSBench, self).train_and_evaluate(model_spec, config, model_dir)

    def query(self, model_spec, epochs=600, stop_halfway=False):
        """
        TODO make this the default one, to reduce excesive amount of hard code.
        :param model_spec:
        :param epochs:
        :param stop_halfway:
        :return:
        """
        if epochs not in self.valid_epochs:
            raise ValueError('invalid number of epochs, must be one of %s'
                             % self.valid_epochs)
        # IPython.embed()
        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
        sampled_index = random.randint(0, len(computed_stat[epochs]) - 1)
        computed_stat = computed_stat[epochs][sampled_index]

        data = {}
        for k in self.fixed_statistics_keys:
            data[k] = fixed_stat[k]
        for k in self.computed_statistics_keys:
            data[k] = computed_stat[k]

        # self.training_time_spent += data['training_time']
        if stop_halfway:
            self.total_epochs_spent += epochs // 2
        else:
            self.total_epochs_spent += epochs

        return data

    def _perf_fn(self, data):
        return data['final_test_accuracy']
    
    def _loss_fn(self, data):
        return 1 - data['final_test_accuracy'] / 100

