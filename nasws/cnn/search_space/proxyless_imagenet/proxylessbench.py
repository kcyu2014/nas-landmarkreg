#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/10/15 上午10:52
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

import IPython
import numpy as np

import utils
from ..api import BenchmarkDatasetTemplate
from ..api import OpSpecTemplate, CellSpecTemplate


RANDOM = '/home/yukaiche/pycharm/automl/experiments/imagenet-search/random_policy*/random_net'
PROXYLESSNAS_MOBILE = '/home/yukaiche/pycharm/automl/experiments/imagenet-search/proxylessnas-*-gpu4/learned_net-fp32'


# make use of config. from proxylessNas, as well as config generation.
class ProxylessNasBlockSpec(OpSpecTemplate):
    # should faithfully represent the operations.
    typename = 'ProxylessNasOpSpec'
    cls_name = 'MobileInvertedResidualBlock'
    op_choices = {
        'conv_candidates': ['3x3_MBConv3', '3x3_MBConv6','5x5_MBConv3', '5x5_MBConv6', '7x7_MBConv3', '7x7_MBConv6', 'Zero'],
    }
    active_index = [0]
    auxilary_node = {}
    output_node_number = 0 # no input no output, just the inner stuff.

    def __init__(self, ops):
        """ ops == [active_index, ... ] for each block. """
        super(ProxylessNasBlockSpec, self).__init__(ops, 'List')

    def check_consistency(self, matrix, single_op_per_node=False):
        return True

    @staticmethod
    def active_index_from_block_config(config):
        """ return the active index of MobileInvertedResidualBlock, and set this accordingly. """
        assert config['name'] == 'MobileInvertedResidualBlock'
        layer_config = config['mobile_inverted_conv']
        if layer_config['name'] == 'MBInvertedConvLayer':
            return ProxylessNasBlockSpec.op_choices['conv_candidates'].index(
                '{a}x{a}_MBConv{b}'.format(a=layer_config['kernel_size'], b=layer_config['expand_ratio']))
        elif layer_config['name'] == 'ZeroLayer':
            return 6
        else:
            IPython.embed(header='check what s wrong here. ')
            raise ValueError("ProxylessNAS only have search space with MBInvertedConvLayer.")

    @classmethod
    def build_op_from_list(cls, input_list):
        """ input list is a block configs """
        if input_list[0]['mobile_inverted_conv']['expand_ratio'] == 1:
            # removing the first block.
            input_list = input_list[1:]

        return cls([ProxylessNasBlockSpec.active_index_from_block_config(config) for config in input_list])

    def labeling(self):
        return self.ops

    def __str__(self):
        return str((self.op_choices['conv_candidates'][i] for i in self.ops))


class ProxylessNasModelSpec(CellSpecTemplate):
    
    def __init__(self, matrix, ops:ProxylessNasBlockSpec, data_format='channel_last'):
        len_ops = len(ops)
        matrix = np.eye(len_ops, k=1)
        self.net_config = None
        super(ProxylessNasModelSpec, self).__init__(matrix, ops, data_format)

    @ classmethod
    def build_from_net_config(cls, net_config):
        IPython.embed(header='hhh')
        spec = cls(None, ProxylessNasBlockSpec.build_op_from_list(net_config['blocks']))
        spec.net_config = net_config
        return spec


class ProxylessNasBench(BenchmarkDatasetTemplate):

    fixed_statistics_keys = ['labeling', 'trainable_parameters', 'flops', 'net_config', 'meta_information']
    computed_statistics_keys = ['final_train_accuracy', 'final_validation_accuracy','final_test_accuracy',
                                'final_train_loss', 'final_validation_loss','final_test_loss',
                                'final_validation_accuracy_5','final_test_accuracy_5',]

    def __init__(self, dataset_file, model_arch_file='dartsbench_model_specs.json', seed=None, config=None):
        self.dataset_file = dataset_file
        self.valid_epochs = {120}
        super(ProxylessNasBench, self).__init__(dataset_file, seed, config)

    def load_dataset_file(self, dataset_file):
        logging.info("Loading from the disk")
        self.preprocess_dataset_from_given_files()

    def update_statistics_by_metadata(self, model_spec: ProxylessNasModelSpec, metadata, epoch=120):
        hash = model_spec.hash_spec()
        fix_stat_values = [model_spec.ops,] + [metadata[k] for k in self.fixed_statistics_keys[1:]]
        eval_results = metadata['evaluation_results'][-1]
        compute_stat_values = [eval_results[k.split('final_')[1]] for k in self.computed_statistics_keys]
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
        if preload_from_folder:
            p = preload_from_folder
            seed = int(p.split('-')[2])
            output = utils.load_json(p + '/output')
            net_config = utils.load_json(p + '/net.config')
            net_info = utils.load_json(p + '/logs/net_info.txt')
            metadata = {
                 'evaluation_results': [{
                    'training_time': 0,
                    'train_accuracy': 0,
                    'validation_loss': output['valid_loss'],
                    'validation_accuracy': output['valid_acc1'],
                    'validation_accuracy_5': output['valid_acc5'],
                    'test_loss': output['test_loss'],
                    'test_accuracy': output['test_acc1'],
                    'test_accuracy_5': output['test_acc5'],
                }],
                'trainable_params': net_info['param'],
                'flops': net_info['flops'],
                'seed': seed,
                'net_config': net_config
            }
            return metadata
        else:
            raise ValueError("Please use the 'imagenet_run_exp' to run the thing. ")

    def preprocess_dataset_from_given_files(self):
        import glob
        # random paths
        for method, path in zip(['random', 'proxylessnas_mobile_gradient'], [RANDOM, PROXYLESSNAS_MOBILE]):
            for p in glob.glob(path):
                metadata = self.train_and_evaluate(None, None, None, p)
                net_config = metadata['net_config']
                metadata['meta_information'] = {'approach': method, 'path': p}
                model_spec = ProxylessNasModelSpec.build_from_net_config(net_config)
                self.update_statistics_by_metadata(model_spec, metadata)

        logging.info("Loaded train model from give paths.")
