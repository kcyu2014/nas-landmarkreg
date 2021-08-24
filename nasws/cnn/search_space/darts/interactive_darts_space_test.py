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


# %%
