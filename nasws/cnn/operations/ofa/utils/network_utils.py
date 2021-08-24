import numpy as np
import os
import sys
import math
import torch

try:
    import horovod.torch as hvd
except ImportError:
    # print('No horovod in environment')
    import numpy as hvd
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list


def list_sum(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] + list_sum(x[1:])


def list_weighted_sum(x, weights):
    if len(x) == 1:
        return x[0] * weights[0]
    else:
        return x[0] * weights[0] + list_weighted_sum(x[1:], weights[1:])


def list_mean(x):
    return list_sum(x) / len(x)


def list_mul(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] * list_mul(x[1:])


def list_join(val_list, sep='\t'):
    return sep.join([
        str(val) for val in val_list
    ])


def list_continuous_index(val_list, index):
    assert index <= len(val_list) - 1
    left_id = int(index)
    right_id = int(math.ceil(index))
    if left_id == right_id:
        return val_list[left_id]
    else:
        return val_list[left_id] * (right_id - index) + val_list[right_id] * (index - left_id)


def subset_mean(val_list, sub_indexes):
    sub_indexes = int2list(sub_indexes, 1)
    return list_mean([val_list[idx] for idx in sub_indexes])


def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def int2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def download_url(url, model_dir='~/.torch/', overwrite=False):
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    cached_file = model_dir
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file

