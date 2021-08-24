import glob
from collections import defaultdict
from nasws.cnn.visualization import tensorboard_check_tags, tensorboard_load_summary
from utils import load_args, load_json, count_parameters_in_MB
from nni.nas.pytorch.fixed import FixedArchitecture
import torch
from thop import profile
import numpy as np


def average_last_K(l, top_K=5):
    return sum(l[-top_K:]) / top_K

def collect_experiment_kdt_from_tensorboard(path):
    args = load_args(path + '/args.json')
    print(args)

    # store all the results as follow
    tb_paths = glob.glob(path + '/log/*')
    res = defaultdict()

    for p in tb_paths:
        # print(p)
        tags = tensorboard_check_tags(p)
        for t in tags:
            steps, r = tensorboard_load_summary(p, t)
            if t in res:
                res[t] += list(zip(steps, r))
            else:
                res[t] = list(zip(steps, r))

    tag_specified = [
        'validation/sparse_kdt_0.0001', 
        'validation/sparse_spr_0.0001']
    final_res = {}
    for tag in tag_specified:
        d = sort_tb_pairs(res[tag])
        final_res[tag] = average_last_K(d)
    
    return final_res



def collect_experiment_result(path):
    # the final evaluation model should be recomputed based on the results over server
    # load args
    args = load_args(path + '/args.json')
    # print(args)

    # store all the results as follow
    tb_paths = glob.glob(path + '/log/*')
    res = defaultdict()

    for p in tb_paths:
        # print(p)
        tags = tensorboard_check_tags(p)
        for t in tags:
            steps, r = tensorboard_load_summary(p, t)
            if t in res:
                res[t] += list(zip(steps, r))
            else:
                res[t] = list(zip(steps, r))
        

    # print(res.keys())
    # collect the associated statistics
    num_epoch = len(res['train/sum'])
    num_channels = 256 # fixed across the entire dataset
    num_cells = 4

    seed = 0

    # store all the intermediate results of 1 run.
    all_train_loss = sort_tb_pairs(res['train/sum'])
    all_valid_loss = sort_tb_pairs(res['validation/ReDWeb'])

    train_loss = average_last_K(sort_tb_pairs(res['train/sum']))
    valid_loss = average_last_K(sort_tb_pairs(res['validation/ReDWeb']))

    # from the current log, this is at is. we do not have more to analyze
    # From this point, we need to get the result from checkpoint and store all the statistics accordingly
    # use this to directly apply 

    arch = load_json(path + '/arch.json')
    print('processing architecture ',arch)

    # model = MidasNetSearch(backbone='resnext101_wsl', args=args)
    # mutator = FixedArchitecture(model, arch)
    # mutator.reset() 

    # ckpt_path = path + '/checkpoint.pt'
    # if os.path.exists(ckpt_path):
    #     print('loading checkpoint...')
    #     checkpoint = torch.load(ckpt_path)
    #     model.load_state_dict(checkpoint['model'])
    # print('finish loading the model ...')

    # count parameters
    # num_param = count_parameters_in_MB(model)
    num_param = 0

    return num_epoch, train_loss, valid_loss, num_param, arch, all_train_loss, all_valid_loss

