#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/20 下午3:38
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================
import logging
import os

import IPython
import time

import utils
from ..api import BenchmarkDatasetTemplate
from ..darts import DartsNetworkImageNetSearch, DartsNetworkCIFARSearch, \
    NetworkCIFAR, NetworkImageNet, DartsSearchSpace, DartsModelSpec, Genotype
from ..darts.nni.nni_imagenet import query_nds_trial_stats

NDS_FIXED_KEYS = ['job_id', 'exp_mem', 'params', 'flops',
                  'net', 'optim', 'act_mem','prec_time', 'iter_time', 'min_test_top1']

NDS_COMPUTE_KEYS = ['train_ep_loss', 'train_ep_top1',
                    'test_ep_top1', 'train_it_loss', 'train_it_top1']

def process_nds_entry(entry, epoch_limit):
    """
    Computed stat organization
     # epochs --> repeat index --> metric name --> scalar
    Fixed stat organization
     # metric name --> scalar

    Keys
    'job_id', 'rng_seed', 'exp_mem', 'params', 'flops', 'net', 'optim', 'act_mem',
        'prec_time', 'iter_time', 'min_test_top1',
        'train_ep_loss', 'train_ep_top1', 'test_ep_top1', 'train_it_loss', 'train_it_top1'
    :param entry: each column loaded in json with keys.
    :return: statistics (computed)
    """
    # fixed statistics (all the rest)
    # computed statistics. (final_train/valid/test_accuracy)
    genotype = entry['net']['genotype']
    rpt_index = entry['rng_seed']
    fixed_stat_keys = NDS_FIXED_KEYS
    compute_stat_keys = NDS_COMPUTE_KEYS

    fixed_stat = {k: entry[k] for k in fixed_stat_keys if k in entry.keys()}
    fixed_stat['genotype'] = genotype

    computed_stat = {}
    for ep in range(epoch_limit):
        stat = {}
        for k in compute_stat_keys:
            try:
                stat[k] = entry[k][ep]
            except IndexError:
                continue
        computed_stat[ep] = {int(rpt_index): stat}

    return Genotype(genotype['normal'], genotype['normal_concat'], genotype['reduce'], genotype['reduce_concat']), \
           fixed_stat, computed_stat


class NDSDataset(BenchmarkDatasetTemplate):
    
    available_settings = ['Amoeba_in', 'PNAS', 'PNAS_in', 'Vanilla_lr-wd_in',
                          'ResNeXt-B', 'ResNeXt-A',
                          'DARTS_in', 'ResNeXt-B_in',
                          'NASNet', 'ResNet_reruns', 'ResNet_rng1',
                          'DARTS_lr-wd', 'Vanilla_reruns', 'ENAS_fix-w-d',
                          'ResNet_lr-wd', 'Vanilla', 'DARTS',
                          'ResNet_lr-wd_in', 'Amoeba',
                          'Vanilla_rng3', 'ResNet_rng3', 'ResNet_rng2',
                          'Vanilla_rng2', 'Vanilla_rng1',
                          'ENAS', 'PNAS_fix-w-d', 'ResNeXt-A_in', 'ResNet-B',
                          'Vanilla_lr-wd', 'ENAS_in', 'ResNet', 'DARTS_fix-w-d', 'NASNet_in', 'DARTS_lr-wd_in']
    fixed_statistics_keys = ['job_id', 'exp_mem', 'params', 'flops',
                       'net', 'optim', 'act_mem','prec_time', 'iter_time', 'min_test_top1']
    computed_statistics_keys = ['train_ep_loss', 'train_ep_top1',
                         'test_ep_top1', 'train_it_loss', 'train_it_top1']

    # training logistics are kept for 100 epochs.
    # valid_epochs = set([str(i) for i in range(100)])

    perf_fn = lambda data: data['test_ep_top1']

    def __init__(self, dataset_file, seed=None, config=None, epoch_limit=100):
        self.dataset_dir = dataset_file
        self.dataset_file = f'{dataset_file}/{config}-processed.json'
        self.config = config
        self.epoch_limit = epoch_limit
        self.valid_epochs = list([i for i in range(epoch_limit)])
        self.load_dataset_file(self.dataset_file)

    def create_model_spec(self, fixed_data):
        # IPython.embed()
        genotype = fixed_data['genotype']
        g = Genotype(genotype['normal'], genotype['normal_concat'], genotype['reduce'], genotype['reduce_concat'])
        return DartsModelSpec.from_darts_genotype(g)

    def preprocess_dataset_from_given_files(self):
        """ Load the specific dataset file """
        if not self.config in self.available_settings:
            raise ValueError("Config not supported.")
        begintime = time.time()
        print(f'epoch number limit : {self.epoch_limit}')
        d = utils.load_json(f"{self.dataset_dir}/{self.config}.json")
        for entry in d:
            genotype, fixed_stat, computed_stat = process_nds_entry(entry, self.epoch_limit)
            spec = DartsModelSpec.from_darts_genotype(genotype)
            _h = spec.hash_spec()
            if _h not in self.hash_dict.keys():
                self.hash_dict[_h] = spec
                self.fixed_statistics[_h] = fixed_stat
                self.computed_statistics[_h] = computed_stat
            else:
                rpt_index = list(computed_stat[0].keys())[0]
                for ep in range(self.epoch_limit):
                    self.computed_statistics[_h][ep][rpt_index] = computed_stat[ep][rpt_index]
        endtime = time.time()
        logging.info("Total Model Loaded {} using {} seconds".format(len(self.hash_dict.keys()), endtime-begintime))

    def query(self, model_spec, epochs=None, stop_halfway=False):
        epochs = epochs or self.epoch_limit - 1
        return super(NDSDataset, self).query(model_spec, epochs, stop_halfway)

    def query_perf(self, model_spec):
        return 100 - self.query(model_spec)['test_ep_top1']

    def query_loss(self, model_spec):
        return 



def generate_nni_dataset_file_name(dataset, model_family, proposer):
    model_arch_file = f'nni_modelspecs_{dataset}_{proposer}_{model_family}.json'
    dataset_path = f'nni_dataset_{dataset}_{proposer}_{model_family}.json'
    return model_arch_file, dataset_path


def process_nds_nni_entry(entry):
    """
    Computed stat organization
     # epochs --> repeat index --> metric name --> scalar
    Fixed stat organization
     # metric name --> scalar

    Keys
    'job_id', 'rng_seed', 'exp_mem', 'params', 'flops', 'net', 'optim', 'act_mem',
        'prec_time', 'iter_time', 'min_test_top1',
        'train_ep_loss', 'train_ep_top1', 'test_ep_top1', 'train_it_loss', 'train_it_top1'
    :param entry: each column loaded in json with keys.
    :return: statistics (computed)
    """
    # fixed statistics (all the rest)
    # computed statistics. (final_train/valid/test_accurac
    # develop here, to process a nni entry as normal
    genotype = entry['net']['genotype']
    rpt_index = entry['rng_seed']
    fixed_stat_keys = NDS_FIXED_KEYS
    compute_stat_keys = NDS_COMPUTE_KEYS

    fixed_stat = {k: entry[k] for k in fixed_stat_keys}
    fixed_stat['genotype'] = genotype

    computed_stat = {}
    for ep in range(100):
        stat = {}
        for k in compute_stat_keys:
            try:
                stat[k] = entry[k][ep]
            except IndexError:
                continue
        computed_stat[ep] = {int(rpt_index): stat}

    return Genotype(genotype['normal'], genotype['normal_concat'], genotype['reduce'], genotype['reduce_concat']), \
           fixed_stat, computed_stat


class NDSDatasetNNI(NDSDataset):
    """NNI Query wrapper to this particular project.

    Parameters
    ----------
    BenchmarkDatasetTemplate : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    dataset_choices = ['cifar10', 'imagenet']
    model_families = ['nas_cell', 'residual_bottleneck', 'residual_basic', 'vanilla']
                             
    fixed_statistics_keys = ['job_id', 'exp_mem', 'params', 'flops',
                       'net', 'optim', 'act_mem','prec_time', 'iter_time', 'min_test_top1']
    
    computed_statistics_keys = ['train_ep_loss', 'train_ep_top1',
                         'test_ep_top1', 'train_it_loss', 'train_it_top1']

    # training logistics are kept for 100 epochs.

    perf_fn = lambda data: data['test_ep_top1']

    def __init__(self, data_dir, dataset, model_family, proposer='darts'):
        d_file, m_file = generate_nni_dataset_file_name(dataset, model_family, proposer)
        self.dataset_file = os.path.join(data_dir, d_file)
        self.model_arch_file = os.path.join(data_dir, m_file)
        self.config = {'model_family': model_family, 'dataset': dataset, 'proposer': proposer, 'generator': 'random'}

        # TODO update the epochs here!
        self.valid_epochs = {'imagenet': None, 'cifar10': None}[dataset]
        super(NDSDatasetNNI, self).__init__(self.dataset_file, 0, None)

    def create_model_spec(self, fixed_data):
        genotype = fixed_data['genotype']
        g = Genotype(genotype['normal'], genotype['normal_concat'], genotype['reduce'], genotype['reduce_concat'])
        return DartsModelSpec.from_darts_genotype(g)

    def preprocess_dataset_from_given_files(self):
        """ Load the specific dataset file """
        begintime = time.time()
        gen = query_nds_trial_stats(**self.config)

        for ind, entry in enumerate(gen):
            if ind > 10:
                break
    
            genotype, fixed_stat, computed_stat = process_nds_entry(entry)
            spec = DartsModelSpec.from_darts_genotype(genotype)
            _h = spec.hash_spec()

            if _h not in self.hash_dict.keys():
                self.hash_dict[_h] = spec
                self.fixed_statistics[_h] = fixed_stat
                self.computed_statistics[_h] = computed_stat
            else:
                rpt_index = list(computed_stat[0].keys())[0]
                for ep in range(100):
                    self.computed_statistics[_h][ep][rpt_index] = computed_stat[ep][rpt_index]
        endtime = time.time()
        logging.info("Total Model Loaded {} using {} seconds".format(len(self.hash_dict.keys()), endtime-begintime))

    def query(self, model_spec, epochs=99, stop_halfway=False):
        # epochs = str(epochs)
        return super(NDSDataset, self).query(model_spec, epochs, stop_halfway)

    def query_perf(self, model_spec):
        return 100 - self.query(model_spec)['test_ep_top1']

    def query_loss(self, model_spec):
        return 


class DARTSSearchSpaceNDS(DartsSearchSpace):

    def __init__(self, args, full_dataset=False):
        super(DartsSearchSpace, self).__init__(args)
        self.topology_fn = NetworkCIFAR if 'cifar' in args.dataset else NetworkImageNet
        self.model_fn = DartsNetworkCIFARSearch if 'cifar' in args.dataset else DartsNetworkImageNetSearch
        if args.dataset == 'imagenet':
            self.dataset = NDSDataset(os.path.join(self.args.data, 'nds_data'), config='DARTS_in', epoch_limit=50)
            self.dataset.fixed_statistics_keys = ['job_id', 'exp_mem', 'params', 'flops',
                    'net', 'optim', 'act_mem', 'iter_time', 'min_test_top1']
            self.top_K_complete_evaluate = 50
        else:
            self.dataset = NDSDataset(os.path.join(self.args.data, 'nds_data'), config='DARTS')
        self._construct_search_space()
        

class ENASSearchSpaceNDS(DartsSearchSpace):

    def __init__(self, args, full_dataset=False):
        super(DartsSearchSpace, self).__init__(args)
        self.dataset = NDSDataset(os.path.join(self.args.data, 'nds_data'), config='ENAS')
        self._construct_search_space()


class NASNetSearchSpaceNDS(DartsSearchSpace):

    def __init__(self, args, full_dataset=False):
        super(DartsSearchSpace, self).__init__(args)
        self.dataset = NDSDataset(os.path.join(self.args.data, 'nds_data'), config='NASNet')
        self._construct_search_space()


# class ImagenetSearchSpaceNDS(DartsSearchSpace):

#     def __init__(self, args, full_dataset=False):
#         super(DartsSearchSpace, self).__init__(args)
#         self.topology_fn = NetworkImageNet
#         self.model_fn = DartsNetworkImageNetSearch
#         self.dataset = NDSDataset(os.path.join(self.args.data, 'nds_data'), config='DARTS_in', epoch_limit=50)
#         self.dataset.fixed_statistics_keys = ['job_id', 'exp_mem', 'params', 'flops',
#                     'net', 'optim', 'act_mem', 'iter_time', 'min_test_top1']
#         # self.dataset = NDSDatasetNNI(args.data, 'imagenet', 'nas_cell', 'darts')
#         self._construct_search_space()
