import os
from .cifar10 import model_loader as cld

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cld.load(model_name, model_file, data_parallel)
    return net
