#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/26 下午4:29
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================

#%%
# init
# used everywhere.
from IPython import get_ipython
from torch.optim import SGD
if get_ipython() is not None:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().system('nvidia-smi')

from argparse import Namespace
import sys
import os

home = os.environ['HOME']
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print(os.environ['CUDA_VISIBLE_DEVICES'])
os.chdir(f'{home}/pycharm/automl')
# os.chdir(f'{home}/pycharm/automl/nasws/rnn')
sys.path.append(f'{home}/pycharm/nasbench')
sys.path.append(f'{home}/pycharm/automl')
import torch
import torch.nn as nn
from utils import get_logger
logger = get_logger("nasbench201-test")


# %%

from nasws.cnn.search_space.nasbench201.nasbench102_search_space import NASBench201SearchSpace
args = Namespace(data='/Users/yukaiche/pycharm/data', debug=True, landmark_sample_method='fixed',landmark_loss_weighted=False)
space = NASBench201SearchSpace(args)

#%%
print(space.random_topology())

# %%
# debugging here .
from nasws.cnn.search_space.nasbench201.lib.aa_nas_api import AANASBenchAPI
nasbench = AANASBenchAPI('../data/nasbench-102/nasbench201-v1.pth')

# %%
# print(nasbench.meta_archs[:10])
nasbench.query_by_arch(nasbench.meta_archs[0])
# %%

info = nasbench.arch2infos[0]
info.get_dataset_names()

# %%
print(info)

# %%
type(nasbench.meta_archs[0])

# %%
from nasws.cnn.search_space.nasbench201.nasbench102_dataset import NASBench201Benchmark

dataset = NASBench201Benchmark('data/nasbench-102/nasbench201-v1-cifar10.json', full_datset=True)

#%%
# dataset.valid_epochs = [200]
for ind,(k, v) in enumerate(dataset.hash_dict.items()):
    if ind > 1:
        break
    print(dataset.query_perf(v))
    # print(dataset.computed_statistics[k]['200']['888'].keys())

#%%
dataset.perf_fn = lambda data: data['test_acc']
print(dataset.perf_fn(a))


# %%
import utils
d = utils.load_json(dataset.dataset_file)

#%%
print(d.keys())
print(d['compute_stat']['|skip_connect~0|+|skip_connect~0|skip_connect~1|+|none~0|none~1|nor_conv_1x1~2|'].keys())
# %%

#%%

# Doing the mutate-topology function here.

from nasws.cnn.search_space.nasbench201.nasbench102_search_space import mutate_arch_func, random_architecture_func
from nasws.cnn.search_space.nasbench201.lib.models import get_search_spaces

_, arch = space.random_topology()
op_choices = get_search_spaces('cell', 'aa-nas')
rand_arch_generator = random_architecture_func(4, op_choices)

arch2 = rand_arch_generator()

mutate_topology = mutate_arch_func(op_choices)
new_arch = mutate_topology(arch)
new_arch2 = mutate_topology(arch2)
print(arch)
print(new_arch)

print(arch2)
print(new_arch2)
n_arch = space.mutate_topology(arch)
print(n_arch)
print(space.mutate_topology(n_arch))

# finish testing the nasbench201 mutate topology as original.
# %%
