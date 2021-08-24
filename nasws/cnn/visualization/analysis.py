from argparse import Namespace
import sys
import os
import numpy as np


# Data analysis.
from nasws.cnn.search_space.nasbench201.nasbench201_search_space import NASBench201Benchmark, NASBench201SearchSpace
from nasws.cnn.search_space.nasbench101.nasbench_search_space import NASBench_v2, NASbenchSearchSpace, NasBenchSearchSpaceFixChannels, NasBenchSearchSpaceSubsample
from nasws.cnn.search_space.nds.nds_search_space import DARTSSearchSpaceNDS
from nasws.cnn.policy.cnn_search_configs import build_default_args
from nasws.cnn.utils import Rank, compute_percentile, compute_sparse_kendalltau, sort_hash_perfs
from .tool import spares_kdt_compute_mean_std, prob_surpass_random, MAXRANK, load_data_from_experiment_root_dir, FILTER_LOW_ACC


# load the all these nasbench here, for query the final performance.


def get_space(name):
    args = build_default_args()
    if name == 'nasbench101':
        args.num_intermediate_nodes = 5
        space = NASbenchSearchSpace(args)
    elif name == 'nasbench101_fix_channels':
        space = NasBenchSearchSpaceFixChannels(args)
    elif name == 'nasbench101_subsample':
        space = NasBenchSearchSpaceSubsample(args)
    elif  'nasbench201' in name:
        args.num_intermediate_nodes = 4
        space = NASBench201SearchSpace(args)
    elif 'darts_nds' in name:
        space = DARTSSearchSpaceNDS(args)
    else:
        raise NotImplementedError("Not yet supported")
    return space


def initialize_space_data_dict():
        #initialize space data dict
    space_data_dict = {
        'nasbench101': {},
        'nasbench201': {},
        'darts_nds': {}
    }

    keys = ['acc', 'perf', 'acc-std',  'perf-std', 'search_ids', 'search_perfs', 'args', 'p-random', 'res_dicts', 
            'kdt', 'kdt-std', 'skdt','skdt-std', 
            'spr', 'spr-std', 'sspr', 'sspr-std']
    for s in space_data_dict.values():
        for k in keys:
            s[k] = []

    return space_data_dict


def add_supernetacc_finalperf_kdt(_res_dicts, search_spaces, space_data_dict, 
                                  final_result=False, threshold=0.4, use_hash=False, sort_best_model_fn=None):
    """Analysis the supernet accordingly.
    
    Parameters
    ----------
    _res_dicts : [type]
        [description]
    search_spaces : [type]
        [description]
    space_data_dict : dict
        dict[spacename][dataname]
        spacename as "nasbench101" etc
        dataname: ['acc', 'kdt', 'perf', 'acc-std', 'kdt-std', 'perf-std']
    
    Returns
    -------
    space_data_dict
    """
    

    def data_fn(res):
        # ouput the data
        return None
    args = _res_dicts[0]['args']
    steps = list(sorted(list(_res_dicts[0]['eval_epochs'])))
    TOP_K = 10 if args.search_space == 'nasbench201' else 5
    # TOP_K = 3
    # this is fixed
    # .
    if final_result:
        _steps = steps[-1:]
    else:
        _steps = steps
    
    for s in _steps:
        # print("step running here", s)    
        r = spares_kdt_compute_mean_std(_res_dicts, search_spaces, 'eval_arch_' + str(s), 'eval_perf_' + str(s), use_hash=use_hash, sort_best_model_fn=sort_best_model_fn)
        arch_ids = r[5]
        arch_supernet_perfs = r[4]
        s_arch_ids, _ = sort_hash_perfs(arch_ids, arch_supernet_perfs)
        search_arch_ids = s_arch_ids[-TOP_K:]

        gt_perfs = search_spaces[args.search_space].query_gt_perfs(search_arch_ids)
        gt_std = np.std(gt_perfs)
        ws_perf = np.mean(arch_supernet_perfs)
        ws_std = np.std(arch_supernet_perfs)
        if ws_perf < threshold:
            if FILTER_LOW_ACC:
                print("Data is not trained, filtered!")
                continue
        p_r = prob_surpass_random(max(search_arch_ids), MAXRANK[args.search_space], repeat=TOP_K)
        #  skdt, sspr, kdt, spr,
        space_data_dict[args.search_space]['args'].append(args)
        space_data_dict[args.search_space]['acc'].append(ws_perf)
        space_data_dict[args.search_space]['acc-std'].append(ws_std)
        space_data_dict[args.search_space]['skdt'].append(r[0][0])
        space_data_dict[args.search_space]['skdt-std'].append(r[0][1])
        space_data_dict[args.search_space]['kdt'].append(r[2][0])
        space_data_dict[args.search_space]['kdt-std'].append(r[2][1])
        space_data_dict[args.search_space]['spr'].append(r[3][0])
        space_data_dict[args.search_space]['spr-std'].append(r[3][1])
        space_data_dict[args.search_space]['sspr'].append(r[1][0])
        space_data_dict[args.search_space]['sspr-std'].append(r[1][1])
        space_data_dict[args.search_space]['perf'].append(np.mean(gt_perfs))
        space_data_dict[args.search_space]['perf-std'].append(gt_std)
        space_data_dict[args.search_space]['search_perfs'].append(arch_supernet_perfs)
        space_data_dict[args.search_space]['search_ids'].append(arch_ids)
        space_data_dict[args.search_space]['p-random'].append(p_r)
        space_data_dict[args.search_space]['res_dicts'].append(_res_dicts)

    return space_data_dict


def print_basic_statistics_for_folder_table(lr_dir, str_filter, space_name,  filter_fn, search_spaces):
    """Statistics to write in the paper.
    
    Parameters
    ----------
    lr_dir : [type]
        [description]
    str_filter : [type]
        [description]
    space_name : [type]
        [description]
    filter_fn : [type]
        [description]
    search_spaces : [type]
        [description]
    """
    accs, kdt, best_models, p_random, res_dicts = load_data_from_experiment_root_dir(lr_dir, str_filter, original_args=True, target_fn=filter_fn)
    space_data_dict = initialize_space_data_dict()

    for k in res_dicts[space_name].keys():
        space_data_dict = add_supernetacc_finalperf_kdt(res_dicts[space_name][k], search_spaces, space_data_dict, final_result=True)
    print([filter_fn(a) for a in space_data_dict[space_name]['args']])
    for k in ['acc','acc-std', 'kdt', 'kdt-std', 'perf', 'perf-std', 'p-random']:
        print(k)
        print(space_data_dict[space_name][k], np.mean(space_data_dict[space_name][k])) 


