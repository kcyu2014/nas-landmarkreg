#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/26 下午4:45
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
import time
import os
from ..api import BenchmarkDatasetTemplate
from .lib.aa_nas_api import AANASBenchAPI
from .lib.models import CellStructure, get_search_spaces

epochs_by_datasets = {'cifar10': 200, 'cifar10-valid': 200,  'cifar100': 200, 'ImageNet16-120': 200}


def process_nasbench201_entry(entry, dataset):
    """
        computed stats.
        # hash --> epochs --> repeat index --> metric name --> scalar
    :param entry:
    :return: Genotype [CellStructure], fixed_stat, computed_stat
    """

    flop, param, latency = entry.get_comput_costs(dataset)
    epoch = epochs_by_datasets[dataset]

    seeds, train_loss, train_acc = entry.get_metrics(dataset, 'train', mean=False)
    if dataset == 'cifar10-valid':
        v_seeds , valid_loss, valid_acc = entry.get_metrics(dataset, 'x-valid', mean=False)
        t_seeds, test__loss, test__acc = v_seeds, valid_loss, valid_acc
    elif dataset == 'cifar10':
        t_seeds, test__loss, test__acc = entry.get_metrics(dataset, 'ori-test', mean=False)
        v_seeds , valid_loss, valid_acc = t_seeds, test__loss, test__acc
    else:
        v_seeds, valid_loss, valid_acc = entry.get_metrics(dataset, 'x-valid', mean=False)
        t_seeds, test__loss, test__acc = entry.get_metrics(dataset, 'x-test', mean=False)

    fixed_stat = {'arch': entry.arch_str, 'flop': flop, 'latency': latency,
                  'param':param, 'arch_index': entry.arch_index}

    compute_stat = {epoch: {
        seed: {} for seed in set(seeds).union(set(v_seeds)).union(set(t_seeds))
    }}
    prefix = ['train', 'valid', 'test']
    for ind, l in enumerate([[seeds, train_loss, train_acc],
                             [v_seeds , valid_loss, valid_acc],
                             [t_seeds, test__loss, test__acc]]):
        for seed, loss, a in zip(*l):
            compute_stat[epoch][seed][prefix[ind] + '_acc'] = a
            compute_stat[epoch][seed][prefix[ind] + '_loss'] = loss

    return fixed_stat, compute_stat


class NASBench201Benchmark(BenchmarkDatasetTemplate):
    fixed_statistics_keys = ['genotype', 'param', 'flop', 'latency', 'arch', 'arch_index']
    computed_statistics_keys = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'test_loss', 'test_acc']
    valid_epochs = [200,]
    available_ops = get_search_spaces('cell', 'aa-nas')

    def create_model_spec(self, fixed_data):
        return CellStructure.str2structure(fixed_data['genotype'])

    def __init__(self, dataset_file, target_dataset='cifar10', full_datset=False):
        self.dataset_file = dataset_file
        self.config = target_dataset
        self.dataset = AANASBenchAPI(os.path.join(os.path.dirname(self.dataset_file), 'nasbench102-v1.pth'))\
            if full_datset else None
        self.load_dataset_file(dataset_file)

    def preprocess_dataset_from_given_files(self):
        logging.info("Pre-processing NASBench201Benchmark Dataset...")
        begintime = time.time()
        if self.dataset is None:
            self.dataset = AANASBenchAPI(os.path.join(os.path.dirname(self.dataset_file), 'nasbench102-v1.pth'))
        # for all possible data here.
        for ind, entry in self.dataset.arch2infos.items():
            fix_stat, compute_stat = process_nasbench201_entry(entry, self.config)
            _h = fix_stat['arch']
            genotype = self.dataset.arch(fix_stat['arch_index'])
            fix_stat['genotype'] = genotype
            spec = self.create_model_spec(fix_stat)
            if _h not in self.hash_dict.keys():
                self.hash_dict[_h] = spec
                self.fixed_statistics[_h] = fix_stat
                self.computed_statistics[_h] = compute_stat
        endtime = time.time()
        logging.info("Total Model Loaded {} using {} seconds".format(len(self.hash_dict.keys()), endtime-begintime))

    def _check_spec(self, model_spec):
        return isinstance(model_spec, CellStructure)

    def _hash_spec(self, model_spec: CellStructure):
        return model_spec.tostr()

    def query(self, model_spec: CellStructure, epochs=200):
        return super(NASBench201Benchmark, self).query(model_spec, epochs)

    def query_perf(self, model_spec):
        return self.query(model_spec)['test_acc']

    def query_loss(self, model_spec):
        return self.query(model_spec)['test_loss']