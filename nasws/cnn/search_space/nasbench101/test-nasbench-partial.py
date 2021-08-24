#%%
from IPython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# %load_ext autoreload
# %autoreload 2
#%%
get_ipython().run_code('!nvidia-smi')
#%%
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

#%%

import utils
from nasws.dataset.image_dataloader import load_dataset
from nasws.cnn.search_space.nasbench101.nasbench_api_v2 import NASBench_v2
from nasws.cnn.search_space.nasbench101.util import load_nasbench_checkpoint, transfer_model_key_to_search_key, \
    nasbench_zero_padding_load_state_dict_pre_hook
from nasws.cnn.procedures.train_search_procedure import darts_model_validation
from nasws.cnn.search_space.nasbench101.model_search import NasBenchNetSearch

#%% Load cifar dataset.

utils.torch_random_seed(10)
print('loading cifar ...')
train_queue, valid_queue, test_queue = load_dataset()
print("loading Nasbench dataset...")
nasbench = NASBench_v2('data/nasbench/nasbench_only108.tfrecord', only_hash=True)

#%%

sanity_check = True

#%%

# load one model and 
MODEL_FOLDER = '/home/yukaiche/pycharm/automl/experiments/reproduce-nasbench/rank_100002-arch_97a390a2cb02fdfbc505f6ac44a37228-eval-runid-0'
CKPT_PATH = MODEL_FOLDER + '/checkpoint.pt'
model_folder = MODEL_FOLDER
ckpt_path = CKPT_PATH

print("load model and test ...")
hash = model_folder.split('arch_')[1].split('-eval')[0]
print('model_hash ', hash)
spec = nasbench.hash_to_model_spec(hash)
print('model spec', spec)
# model, ckpt = load_nasbench_checkpoint(ckpt_path, spec, legacy=True)

from nasws.cnn.policy.cnn_search_configs import build_default_args

# def project the weights 
default_args = build_default_args()
default_args.model_spec = spec
spec = nasbench.hash_to_model_spec(hash)
model, ckpt = load_nasbench_checkpoint(ckpt_path, spec, legacy=True)
model = model.cuda()
model.eval()
model_search = NasBenchNetSearch(args=default_args)
model_search.eval()
model_search = model_search.cuda()
source_dict = model.state_dict()
target_dict = model_search.state_dict()

#%%

# Checking the loaded model state dict.
if sanity_check:
    for ind, k in enumerate(source_dict.keys()):
        if ind > 5:
            break
        print(k, ((source_dict[k] - target_dict[k]).sum()))

trans_dict = dict()

for k in model.state_dict().keys():
    kk = transfer_model_key_to_search_key(k, spec)
    if kk not in model_search.state_dict().keys():
        print('not found ', kk)
        continue
    # if sanity_check:
    #     print(f'map {k}', source_dict[k].size()) 
    #     print(f'to {kk}' ,target_dict[kk].size())

    trans_dict[kk] = source_dict[k]

padded_dict = nasbench_zero_padding_load_state_dict_pre_hook(trans_dict, model_search.state_dict())

#%%

if sanity_check:
    target_dict = model_search.state_dict()
    for ind, k in enumerate(padded_dict.keys()):
        if ind > 5:
            break
        print(k)
        print((padded_dict[k] - target_dict[k]).sum().item())
        print((ckpt['model_state'][k] - target_dict[k]).sum().item())
        # print((padded_dict[k][trans_dict[k].size()[0]] - trans_dict[k]).sum())

#%%

# loaded the padded dict and test the results.
model_search.load_state_dict(padded_dict)
model_search = model_search.change_model_spec(spec)
model_search = model_search.eval()

#%%

res = darts_model_validation(test_queue, model, nn.CrossEntropyLoss(), Namespace(debug=False, report_freq=50))
print('original model evaluation on test split', res)

search_res = darts_model_validation(test_queue, model_search, nn.CrossEntropyLoss(),
                                    Namespace(debug=False, report_freq=50))
print('Reload to NasBenchSearch, evaluation results ', search_res)
# results get worse after padding to original node. Somethings goes wrong.

#%% 

# Further sanity checking, by loading part of the model and continue.

#%%

# print(model.stacks['stack0']['module0'].vertex_ops.keys())
# print(spec.ops)
# m0 = model.stacks['stack0']['module0']
# print([k for k in m0.state_dict().keys() if 'vertex_1' not in k and 'proj' not in k ])
# # print([k for k in m0.state_dict().keys() if 'vertex_1' not in k and 'proj' not in k ])
# load_keys = sorted([transfer_model_key_to_search_key(k, spec) for k in k1])
# train_param_keys = sorted(train_param_keys)
# # for a1, a2 in zip(load_keys, train_param_keys):
# #     print(a1)
# #     print(a2)
# for a1 in load_keys:
#     if a1 not in train_param_keys:
#         print(a1)
# print(len(load_keys), len(train_param_keys))


#%%

model.eval()
model_search.load_state_dict(padded_dict)
model_search.eval()

#%%

# This debugs the entire network's output.
img, lab = iter(test_queue).__next__()
img = img.cuda()
lab = lab.cuda()
bz = 32

with torch.no_grad():
    m1 = model.forward_debug(img[:bz, :, :, :])
    m2 = model_search.forward_debug(img[:bz, :, :, :])
    for ind, (a, b) in enumerate(zip(m1, m2)):
        print(ind, (a - b).sum().item())

#%%

# model_search.stem[1].eval()
# model.stem[1].eval()
# m1stem_out = model.stem(img)
# model_search.stem.load_state_dict(model.stem.state_dict())
# m2stem_out = model_search.stem(img)
# for k, v in model.stem.state_dict().items():
#     print(k, (v - model_search.stem.state_dict()[k]).sum().item())
# print((m1stem_out - m2stem_out).sum().item())


#%%

s1input = m1[0].cuda()
m1stack = model.stacks['stack0']['module0']
m2stack = model_search.stacks['stack0']['module0']
print(m1stack.__class__.__name__)
print(m2stack.__class__.__name__)

print((m1stack(s1input) - m2stack(s1input)).sum().item())

#%%

m1out = m1stack.forward_debug(s1input)
m2out = m2stack.forward_debug(s1input)

#%%

# Cell level, not true after 1 iters
for k, v in m1out.items():
    print(k, (v - m2out[k]).sum().item())

print(m1stack.execution_order.items())

#%%

# vertex level
m1vertex = m1stack.vertex_ops['vertex_1']
m2vertex = m2stack.vertex_ops['vertex_1']

print('vertex_1 difference', (m1vertex([s1input, ]) - m2vertex([s1input, ])).sum().item())
print(m1vertex)
print(m2vertex)

#%%

# inside vertex level, proj_ops and vertex_op
m1_proj = m1vertex.proj_ops[0](s1input)
m1_out = m1vertex.op(m1_proj)

m2_proj = m2vertex.current_proj_ops[0](s1input)
m2_out = m2vertex.current_op(m2_proj)

print('after v1 proj diff', (m1_proj - m2_proj).sum().item())
print('after v1 diff', (m1_out - m2_out).sum().item())

#%%

# proj level
m1projop = m1vertex.proj_ops[0]
m2projop = m2vertex.current_proj_ops[0]
print(m1projop)
print(m2projop)

#%%

print(m1projop[1].training)
print(m2projop.bn.training)

#%%

# m2vertex.train()
model_search.eval()
print(m1projop[1].training)
print(m2projop.bn.training)

