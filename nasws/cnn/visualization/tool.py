"""Tool kit for   submission. Analyze and visualization.

"""
import glob
import os
import utils
from argparse import Namespace
import numpy as np
from nasws.cnn.utils import compute_sparse_kendalltau, compute_sparse_spearmanr, kendalltau, spearmanr, sort_hash_perfs

MAXRANK = {'nasbench101': 423624, 'nasbench201':15625, 'nasbench102':15625, 'darts_nds': 5000,'nasbench': 423624, 'nasbench101_fix_channels':423624}
ROOT=os.path.join(os.environ['HOME'], 'pycharm/automl', 'experiments/icml-hyperparameters/validate-epochs')
FILTER_LOW_ACC = True


def prob_surpass_random(best_rank, max_rank, repeat=10):
    return ((best_rank / max_rank))**repeat

def sort_tb_pairs(l, ignore_index=True):

    slist = list(sorted(l, key=lambda x: x[0]))
    if ignore_index:
        return list(zip(*slist))[1]
    else:
        return slist


def tensorboard_check_tags(path):
    import tensorflow as tf
    summary_iter = tf.train.summary_iterator(path)
    tags = []
    for a in summary_iter:
        for v in a.summary.value:
            if v.tag not in tags:
                tags.append(v.tag)
    return tags


def tensorboard_load_summary(path, tag):
    """ load data for given tag """
    import tensorflow as tf
    summary_iter = tf.train.summary_iterator(path)
    res = []
    steps = []
    for a in summary_iter:    
        for v in a.summary.value:
            if v.tag == tag:
                res.append(v.simple_value)
                steps.append(a.step)
    return steps, res

def load_tensorboard_data(root_path, tags=None):
    import tensorflow as tf
    files = glob.glob(root_path + '/events.out*')
    if len(files) == 0:
        print('No event files found...')
        return None

    tags = tags or tensorboard_check_tags(files[0])
    data = {t:[[], []] for t in tags}
    
    for f in files:
        summary_iter = tf.train.summary_iterator(f)
        for a in summary_iter:
            for v in a.summary.value:
                if v.tag in tags:
                    data[v.tag][0].append(a.step)
                    data[v.tag][1].append(v.simple_value)
    return data


def parse_space_model_id(name, use_hash=False):
    if name in ['nasbench101', 'nasbench', 'nasbench101_fix_channels', 'nasbench101_subspace']:
        if use_hash:
            return lambda x: x.split(', ')[1] 
        else: 
            return lambda x: int(x.split(',')[0])
    elif 'nasbench201' in name or 'nasbench102' in name:
        return lambda x: int(x.split(',')[0])
    elif name == 'darts_nds':
        return lambda x: int(x.split(',')[0])
    else:
        raise ValueError("parse_space_model_id: {}".format(name))


def load_all_arch_perf_files(res_dict, use_hash=False):
    archs_files = {int(p.split('.')[-1]):p for p in glob.glob(res_dict['args'].main_path + '/*/eval_arch_pool.*') if 'perf' not in p}
    perfs_files = {int(p.split('.')[-1]):p for p in glob.glob(res_dict['args'].main_path + '/*/eval_arch_pool.perf.*') if 'perf' in p}

    # process this files
    # res_dict['eval_arch_data'] = {}
    # res_dict['eval_perf_data'] = {}
    epochs = []
    space = res_dict['args'].search_space
    for e, f in archs_files.items():
        res_dict['eval_arch_{}'.format(e)] = utils.load_list(f, parse_space_model_id(space, use_hash))
        epochs.append(e)
    for e, f in perfs_files.items():
        res_dict['eval_perf_{}'.format(e)] = utils.load_list(f, float)
        epochs.append(e)
    final_epoch = int(res_dict['args'].epochs)
    epochs.append(final_epoch)
    res_dict['eval_perf_{}'.format(final_epoch)] = res_dict['eval_perf']
    res_dict['eval_arch_{}'.format(final_epoch)] = res_dict['eval_arch']
    res_dict['eval_epochs'] = set(sorted(epochs))
    return res_dict

# Best Model Performed +- std. Kd-T, P > random search
def load_results_from_given_experiment_dir(dir_path, use_hash=False):
    """Now we can have multiple sub-folders under this main_path...

    Parameters
    ----------
    dir_path : [type]
        [description]
    use_hash : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    print(f'Loading args from {dir_path}')
    args = utils.load_args(dir_path)
    dir_folder = dir_path.split('/')[-2]
    print(f"{args.search_space} with epoch {args.epochs} and seed {args.seed}.")
    print("Path: " + args.main_path)
    try:
        result_path = glob.glob(args.main_path + f'/{dir_folder}/result.json')[0]
        result = utils.load_json(result_path)
        tensorboard_file = glob.glob(args.main_path + f'/runs/{dir_folder}/*events*')[0]
        # tensorboard_file = None
    except IndexError as e:
        print(f'Exception (load_results_from_given_experiment_dir): {e}. cannot find the files under {args.main_path}')
        return args
    try:
        eval_perf = utils.load_list(glob.glob(args.main_path + f'/{dir_folder}/eval_arch_pool.perf')[0], float)
        eval_arch = utils.load_list(glob.glob(args.main_path + f'/{dir_folder}/eval_arch_pool')[0], 
                                    parse_space_model_id(args.search_space, use_hash))
        # eval_arch_raw = utils.load_list(glob.glob(args.main_path + '/*/eval_arch_pool')[0])
        
    except FileNotFoundError as e:
        print(f'Exception (load_results_from_given_experiment_dir): {e}. cannot find the files under {args.main_path}')
    return {'args': args, 'eval_perf': eval_perf, 'eval_arch': eval_arch, 'res': result, 'tf_events': tensorboard_file}


def load_results_from_given_experiment_dir_old(dir_path, use_hash=False):
    """Now we can have multiple sub-folders under this main_path...

    Parameters
    ----------
    dir_path : [type]
        [description]
    use_hash : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    print(f'Loading args from {dir_path}')
    args = utils.load_args(dir_path)
    print(f"{args.search_space} with epoch {args.epochs} and seed {args.seed}.")
    print("Path: " + args.main_path)
    try:
        result_path = glob.glob(dir_path + '/*/result.json')[0]
        result = utils.load_json(result_path)
        tensorboard_file = glob.glob(args.main_path + '/runs/*/*events*')[0]
        # tensorboard_file = None
    except IndexError as e:
        print(f'Exception (load_results_from_given_experiment_dir): {e}. cannot find the files under {args.main_path}')
        return args
    try:
        eval_perf = utils.load_list(glob.glob(args.main_path + '/*/eval_arch_pool.perf')[0], float)
        eval_arch = utils.load_list(glob.glob(args.main_path + '/*/eval_arch_pool')[0], 
                                    parse_space_model_id(args.search_space, use_hash))
        # eval_arch_raw = utils.load_list(glob.glob(args.main_path + '/*/eval_arch_pool')[0])
        
    except FileNotFoundError as e:
        print(f'Exception (load_results_from_given_experiment_dir): {e}. cannot find the files under {args.main_path}')
    return {'args': args, 'eval_perf': eval_perf, 'eval_arch': eval_arch, 'res': result, 'tf_events': tensorboard_file}


def basic_statistics_of_given_folder(args_path, top_k=3, use_hash=False, sort_best_model_fn=None):
    """Query a folder for some basic statistics.
    """
    
    res_dict = load_results_from_given_experiment_dir(args_path, use_hash)
    if isinstance(res_dict, Namespace):
        args = res_dict
        raise ValueError(f"Data missing for {args.search_space} at epoch {args.epochs} with seed {args.seed}")
    
    eval_perf = res_dict['eval_perf']
    eval_arch = res_dict['eval_arch']
    result = res_dict['res']
    args = res_dict['args']
    # load all eval_data
    res_dict = load_all_arch_perf_files(res_dict, use_hash)
    if eval_perf[1] <= eval_perf[0]:
        eval_arch_ids = [a for a in eval_arch[:top_k]]
        if use_hash:
            best_model = sort_best_model_fn(eval_arch_ids)
        else:
            best_model = max(eval_arch_ids)
    else:
        raise ValueError("Evaluation perf file corrupted! Fatal error here.")
    acc = np.mean(np.asanyarray(eval_perf).astype(np.float32))
    if acc < 15:
        print('not trained, evaluation accuracy is below 15')
        if FILTER_LOW_ACC:
            raise ValueError("Best test accuracy < 15, wrong training file.")
    
    # load kdt results
    kdt_ep = []
    for ep in list(result['kendaltau'].keys())[-1:]:
        kdt_ep.append(ep)
    # print(kdt_ep)
    _kdt = np.mean([result['kendaltau'][ep][0] for ep in kdt_ep])
    return acc, _kdt, best_model, res_dict, args


def fn_get_searched_ids(x):
    arch_ids,arch_supernet_perfs = x[5], x[4]
    s_arch_ids, _ = sort_hash_perfs(arch_ids, arch_supernet_perfs)
    return s_arch_ids[-5:], 0


def load_data_from_experiment_root_dirs(all_list, original_args=False, target_fn=None, use_hash=False, sort_best_model_fn=None):
    """Read the experiments into a dictionary...

    Parameters
    ----------
    all_list : list
        list of folder to process.
    original_args : bool, optional
        if true, return the original args., by default False
    target_fn : function, optional
        Input target function to process the keyword, by default lambda x:int(getattr(x, 'epochs'))
    use_hash : bool, optional
        use hash rather than the arch id, used for NB101 sub spaces, by default False
    
    """

    TOP_K = 3

    best_models = {}
    accs = {}
    tr_accs = {}
    kdt = {}
    p_random = {}
    res_dicts = {}
    total_valid_number = 0

    for path in all_list:
        try:
            acc, _kdt, best_model, res_dict, args = basic_statistics_of_given_folder(
                path, top_k=TOP_K, use_hash=use_hash, sort_best_model_fn=sort_best_model_fn)
        except ValueError as e:
            print(f'Exception(load_data_from_experiment_root_dir): {e}')
            continue
        
        try:
            target_key = target_fn(args)
        except ValueError as e:
            print('target_fn Error: ', e)
            continue
    
        space = args.search_space
        # print(space)
        # loop through the dataset
        # initialize
        for a in [best_models, accs, kdt, p_random, res_dicts]:
            if not space in a.keys():
                a[space] = {}
            if target_key not in a[space].keys():
                a[space][target_key] = []
        
        accs[space][target_key].append(acc)
        kdt[space][target_key].append(_kdt)
        best_models[space][target_key].append(best_model)
        res_dicts[space][target_key].append(res_dict)
        p_random[space][target_key].append(prob_surpass_random(best_model, MAXRANK[space], repeat=TOP_K))
        total_valid_number += 1

    print("Total valid number ", total_valid_number)
    # loaded the performance with random sampling. 
    if original_args:
        return accs, kdt, best_models, p_random, res_dicts
    return accs, kdt, best_models, p_random


def load_data_from_experiment_root_dir(path, str_filter='/*/*/*/*/args.json', original_args=False, 
            target_fn=None, use_hash=False, sort_best_model_fn=None):
    """Entry point of almost all experiments reader
    Here we load the full statistics of a given experiment.  

    Parameters
    ----------
    path : str
        Path to the experiment root dir
    str_filter : str, optional
        glob format, read the args.json to locate one experiment, by default '/*/*/*/*/args.json'
    original_args : bool, optional
        if true, return the original args., by default False
    target_fn : function, optional
        Input target function to process the keyword, by default lambdax:int(getattr(x, 'epochs'))
    use_hash : bool, optional
        use hash rather than the arch id, used for NB101 sub spaces, by default False
    
    Returns
    -------
    list
        accs, kdt, best_models, p_random
    """
    
    all_list = glob.glob(path + str_filter)
    return load_data_from_experiment_root_dirs(all_list, str_filter, original_args, 
            target_fn, use_hash, sort_best_model_fn)


def _round_accs(accs):
    t = np.asanyarray(accs).astype(np.float32)
    # print(t)
    tt = np.where(t <= 1., True, False)
    if not np.all(tt):
        t = t / 100
    return t.tolist()
    
    
def _round_percentile(percentile):
    t = np.asanyarray(percentile).astype(np.float32)
    # print(t)
    tt = np.where(t <= 0.5, True, False)
    if not np.all(tt):
        t = 1 - t
    return t.tolist()


def spares_kdt_compute_mean_std(res_dicts, spaces, arch_key='eval_arch', perf_key='eval_perf', use_hash=False, sort_best_model_fn=None):
    """Sparse KDT computation given res_dicts for one run.

    Parameters
    ----------
    res_dicts : list
        dict of running, multiple seed run contains inside
    arch_key : str, optional
        [description], by default 'eval_arch'
    perf_key : str, optional
        [description], by default 'eval_perf'
    
    Returns
    -------
    list [sparse_kdt, sparse_spr, kdt, spr, arch_perf_mean, model_ids]
        of given epoch
    
    Raises
    ------
    ValueError
        [description]
    """
    if not len(res_dicts) > 0:
        raise ValueError("Dict passed in is wrong.", res_dicts)
    
    arch_perf = {}
    
    for res_dict in res_dicts:
        space = res_dict['args'].search_space
        try:
            eval_perf = res_dict[perf_key]
            eval_arch = res_dict[arch_key]
        except KeyError:
            continue
        eval_perf = _round_accs(np.asanyarray(eval_perf).astype(np.float32).tolist())
        for a, p in zip(eval_arch, eval_perf):
            if use_hash:
                mid = sort_best_model_fn([a])
            else:
                mid = int(a)
            if mid not in arch_perf.keys():
                arch_perf[mid] = []
            arch_perf[mid].append(p)
        
    # compute the statistics.
    model_ids = list(sorted(list(arch_perf.keys())))
    arch_perf_mean = [np.mean(arch_perf[m]) for m in model_ids]
    
    gt_perfs = spaces[space].query_gt_perfs(model_ids)
    s_model_ids, _ = sort_hash_perfs(model_ids, arch_perf_mean)
    skdt = compute_sparse_kendalltau(model_ids, arch_perf_mean, gt_perfs, threshold=1e-2, verbose=False)
    sspr = compute_sparse_spearmanr(model_ids, arch_perf_mean, gt_perfs, threshold=1e-2)
    kdt = kendalltau(arch_perf_mean, gt_perfs)
    spr = spearmanr(arch_perf_mean, gt_perfs)
    return skdt, sspr, kdt, spr, arch_perf_mean, s_model_ids



def spares_kdt_compute_mean_std_v2(res_dicts, spaces, arch_key='eval_arch', perf_key='eval_perf', use_hash=False, sort_best_model_fn=None):
    """ Just compute the average sKdT
    Parameters
    ----------
    res_dicts : list
        dict of running, multiple seed run contains inside
    arch_key : str, optional
        [description], by default 'eval_arch'
    perf_key : str, optional
        [description], by default 'eval_perf'
    
    Returns
    -------
    list [sparse_kdt, sparse_spr, kdt, spr, arch_perf_mean, model_ids]
        of given epoch
    
    Raises
    ------
    ValueError
        [description]
    """
    if not len(res_dicts) > 0:
        raise ValueError("Dict passed in is wrong.", res_dicts)
    
    arch_perf = {}
    skdt = []

    for res_dict in res_dicts:
        space = res_dict['args'].search_space
        try:
            eval_perf = res_dict[perf_key]
            eval_arch = res_dict[arch_key]
        except KeyError:
            continue
        
        # process model_ids
        model_ids = []
        for a in eval_arch:
            if use_hash:
                mid = sort_best_model_fn([a])
            else:
                mid = int(a)
            model_ids.append(mid)
        
        eval_perf = _round_accs(np.asanyarray(eval_perf).astype(np.float32).tolist())
        gt_perfs = spaces[space].query_gt_perfs(model_ids)
        _skdt = compute_sparse_kendalltau(model_ids, eval_perf, gt_perfs, threshold=1e-2, verbose=False)
        # sspr = compute_sparse_spearmanr(model_ids, arch_perf_mean, gt_perfs, threshold=1e-2)
        # kdt = kendalltau(arch_perf_mean, gt_perfs)
        # spr = spearmanr(arch_perf_mean, gt_perfs)
        skdt.append(_skdt[0])

    # compute the statistics.
    return np.mean(skdt), np.std(skdt)

def plot_data_x_y(_res_dicts, target_tags):
    """Obtain the data for plotting.

    
    Parameters
    ----------
    _res_dicts : list
        list of res_dict of given experiment runs, multiple seed
    target_tags : list
         TB tags for data query.
    
    Returns
    -------
    dict, dict
        dict of steps[tag] = list(x-values) # usually known as epochs.
        dict of data[tag] = list(y-values)

    """
    target_data = {}
    target_steps = {}

    for t in target_tags:
        # each tag should average the data and get the mean.
        _multi_d = []
        _multi_s = []
        for d in _res_dicts:    
            tf_path = d['tf_events']
            up_steps = d['args'].epochs
            s, _d = tensorboard_load_summary(tf_path, t)
            if len(s) < up_steps - 10: # for those un-trained thing.
                continue
            _multi_d.append(_d)
            _multi_s.append(s)
        try:
            target_steps[t] = np.mean(np.array(_multi_s), axis=0)
        except TypeError as e:
            print("plot_data_x_y exception happens.", e)
            print(_multi_s)
        target_data[t] = np.mean(np.array(_multi_d), axis=0)

    return target_steps, target_data


def plot_data_sparse_kdt_avg(_res_dicts, search_spaces, data_fn, **kwargs):
    """ Only support computing skdt mean and std for now... """
    steps = list(sorted(list(_res_dicts[0]['eval_epochs'])))
    data = []
    error = [] # here is as p-values.
    # this is fixed.
    for s in steps:
        # print("step ", s)
        y, e = spares_kdt_compute_mean_std_v2(_res_dicts, search_spaces, 'eval_arch_' + str(s), 'eval_perf_' + str(s), **kwargs)
        data.append(y)
        error.append(e)
    
    return steps, data, error


def plot_data_sparse_kdt(_res_dicts, search_spaces, data_fn, **kwargs):
    steps = list(sorted(list(_res_dicts[0]['eval_epochs'])))
    data = []
    error = [] # here is as p-values.
    # this is fixed.
    for s in steps:
        # print("step ", s)
        r = spares_kdt_compute_mean_std(_res_dicts, search_spaces, 'eval_arch_' + str(s), 'eval_perf_' + str(s), **kwargs)
        y, e = data_fn(r)
        data.append(y)
        error.append(e)
    return steps, data, error



def _load_train_accs_compute_mean_std(res_dicts):
    if not len(res_dicts) > 0:
        raise ValueError("Dict passed in is wrong.", res_dicts)
    accs = []
    for res_dict in res_dicts:
        # print(res_dict['tf_events'])
        assert 'Train/top_1_acc' in tensorboard_check_tags(res_dict['tf_events']), print(tensorboard_check_tags(res_dict['tf_events']))
        accs.extend(_round_accs(tensorboard_load_summary(res_dict['tf_events'], 'Train/top_1_acc')[1][-5:]))
    return np.mean(accs), np.std(accs)
    
def _load_evals_acc_compute_mean_std(res_dicts):
    if not len(res_dicts) > 0:
        raise ValueError("Dict passed in is wrong.", res_dicts)
    accs = []
    for res_dict in res_dicts:
        eval_perf = res_dict['eval_perf']
        eval_perf = _round_accs(np.asanyarray(eval_perf).astype(np.float32).tolist())
        accs.extend(eval_perf)
    return np.mean(accs), np.std(accs)


def _load_percentile_compute_mean_std(res_dicts):
    if not len(res_dicts) > 0:
        raise ValueError("Dict passed in is wrong.", res_dicts)
    accs = []
    for res_dict in res_dicts:
        # print(res_dict['tf_events'])
        assert '' in tensorboard_check_tags(res_dict['tf_events']), print(tensorboard_check_tags(res_dict['tf_events']))
        accs.extend(_round_percentile(tensorboard_load_summary(res_dict['tf_events'], 'percentile/max_3')[1][-5:]))
    return np.mean(accs), np.std(accs)
    

def _load_kdt_compute_mean_std(res_dicts):
    if not len(res_dicts) > 0:
        raise ValueError("Dict passed in is wrong.", res_dicts)
    kdt = []
    for res_dict in res_dicts:
        assert 'eval_kendall_tau/random_100_mean' in tensorboard_check_tags(res_dict['tf_events'])
        k = 'eval_kendall_tau/random_100_mean'
        kdt.extend(tensorboard_load_summary(res_dict['tf_events'], k)[1][-2:])
    return np.mean(kdt), np.std(kdt), np.max(kdt), np.min(kdt)



def _load_spares_kdt_compute_mean_std(res_dicts):
    if not len(res_dicts) > 0:
        raise ValueError("Dict passed in is wrong.", res_dicts)
    accs = []
    for res_dict in res_dicts:
        eval_perf = res_dict['eval_perf']
        eval_perf = _round_accs(np.asanyarray(eval_perf).astype(np.float32).tolist())
        accs.extend(eval_perf)
    return np.mean(accs), np.std(accs)

def baseline_data():
    """Wrap the baseline data

    """




def validate_epochs_analysis(path, str_filter='/*/*/*/*/args.json', original_args=False):
    
    all_list = glob.glob(path + str_filter)
    
    TOP_K = 3

    best_models = {}
    accs = {}
    tr_accs = {}
    kdt = {}
    p_random = {}
    res_dicts = {}
    total_valid_number = 0

    for path in all_list:
        try:
            acc, _kdt, best_model, res_dict, args = basic_statistics_of_given_folder(path, top_k=TOP_K)
        except ValueError as e:
            print(e)
            continue
        space =  args.search_space
        epoch = int(args.epochs)

        # loop through the dataset
        # initialize
        for a in [best_models, accs, kdt, p_random, res_dicts]:
            if not space in a.keys():
                a[space] = {}
            if epoch not in a[space].keys():
                a[space][epoch] = []
        
        accs[space][epoch].append(acc)
        kdt[space][epoch].append(_kdt)
        best_models[space][epoch].append(best_model)
        res_dicts[space][epoch].append(res_dict)
        p_random[space][epoch].append(prob_surpass_random(best_model, MAXRANK[space], repeat=TOP_K))
        total_valid_number += 1

    print("Total valid number ", total_valid_number)
    # loaded the performance with random sampling. 
    if original_args:
        return accs, kdt, best_models, p_random, res_dicts
    return accs, kdt, best_models, p_random


def valid_epoch_plot_data(
    path=None, str_filter='/*/*/*/*/args.json', data_fn=validate_epochs_analysis, sorted_keys=True, sorted_map_fn=int, spaces=None
    ):

    path = path or ROOT
    accs, kdt, best_models, p_random, res_dicts = data_fn(path, str_filter, True)
    spaces_kdt = {}
    spaces_accs = {}
    spaces_tr_accs = {}
    spaces_pr = {}
    # target_epochs = {'nasbench101': [50, 100, 150, 200], 'nasbench201':[100, 400, 1200,  2000], 'darts_nds':[100,200,400, 1000]}
    for space in accs.keys():
        
        if sorted_keys:
            target_keys = list(sorted([sorted_map_fn(e) for e in accs[space].keys()]))
        else:
            target_keys = accs[space].keys()
        print("Space: ", space, 'target keys :', target_keys)
        kdt_ep = []
        acc_ep = []
        tr_acc_ep = []
        pr_ep = []
        best_ep = []
        for ep in target_keys:
            tr_acc_ep.append(_load_train_accs_compute_mean_std(res_dicts[space][ep]))
            acc_ep.append(_load_evals_acc_compute_mean_std(res_dicts[space][ep]))
            kdt_ep.append(spares_kdt_compute_mean_std(res_dicts[space][ep], spaces))
            # kdt_ep.append(_load_kdt_compute_mean_std(res_dicts[space][ep]))
            pr_ep.append((np.mean(p_random[space][ep])))
            best_ep.append((1 - np.mean(best_models[space][ep]) / MAXRANK[space] ))
        
        spaces_kdt[space] = kdt_ep    
        spaces_accs[space] = acc_ep
        spaces_pr[space] = pr_ep
        spaces_tr_accs[space] = tr_acc_ep
    
    return spaces_kdt, spaces_accs, spaces_tr_accs, spaces_pr


def valid_epoch_latex_table(str_filter='/*/*lr0.025*/*/*/args.json'):
    accs, kdt, best_models, p_random = validate_epochs_analysis(ROOT, str_filter)
    spaces_kdt = {}
    spaces_accs = {}
    spaces_pr = {}
    target_epochs = {'nasbench101': [100, 200, 300, 400], 'nasbench201':[100, 400, 1200,  1800], 'darts_nds':[100, 200, 300, 400]}
    for space in accs.keys():
        print("Space: ", space)
        epochs = list(sorted(accs[space].keys()))
        print("Epoch " + "& ".join([str(ep) for ep in epochs if ep in target_epochs[space]]))
        kdt_ep = []
        acc_ep = []
        pr_ep = []
        best_ep = []
        for ep in target_epochs[space]:
            # kdt_ep.append("{:.3f} $\pm$ {:.3f}".format(np.mean(kdt[space][ep]), np.std(kdt[space][ep])))
            acc_ep.append("{:.2f} $\pm$ {:.2f}".format(np.mean(accs[space][ep]), np.std(accs[space][ep])))
            # pr_ep.append("{:.2f} $\pm$ {:.2f}".format(np.mean(p_random[space][ep]), np.std(p_random[space][ep])))
            
            kdt_ep.append("{:.3f}".format(np.mean(kdt[space][ep])))
            # acc_ep.append("{:.2f}".format(np.mean(accs[space][ep])))
            pr_ep.append("{:.2f}".format(np.mean(p_random[space][ep])))
            best_ep.append("{:.3f}".format(1 - np.mean(best_models[space][ep]) / MAXRANK[space] ))
        spaces_kdt[space] = kdt_ep    
        spaces_accs[space] = acc_ep
        spaces_pr[space] = pr_ep

    # generate the entire table.
    total_lengths = [len(v) for v in target_epochs.values()]
    spaces = list(target_epochs.keys())

    # generate the tabular header. 
    tabular_header = 'l'
    cmidrule = ""
    cstart = 2
    header = ''
    output_data = ['Epochs ', 'Kd-T ', 'P > R ', 'Acc. '] 
    for space, epochs in target_epochs.items():
        l = len(epochs)
        tabular_header += "c"*l + " c "
        cmidrule += '\cmidrule{' + '{}-{}'.format(cstart, cstart + l-1) + '}'
        cstart += l +1
        header += '&\multicolumn{' + str(l) + '}{c}{' + space.replace('_', '-') + "} &"
        output_data[0] += "& " + " & ".join([str(ep) for ep in epochs]) + "&"
        output_data[1] += "& " + " & ".join(spaces_kdt[space])+ " & "
        output_data[2] += "& " + " & ".join(spaces_pr[space])+ " & "
        output_data[3] += "& " + " & ".join(spaces_accs[space])+ " & "

    print("\\begin{"+ 'tabular' +"}", '{', tabular_header , '}')
    print("\\toprule")
    print(header + '\\\\')
    print(cmidrule)
    for d in output_data:
        print(d + '\\\\')
    # generate Cmidrule


def darts_nds_map_state_dict_keys_from_non_wsbn_to_wsbn(state_dict):
    """Mapping the DARTS NDS old state_dict into the new version

    To save time of the re-training. Used to counter API changes after introduce WSBN layer into DARTS_NDS.

    
    Parameters
    ----------
    state_dict : [Model dict state] name with old implementation
    
    Return
    --------
    new_state_dict: model state dict with new keys.
    """
    def _replace_dict_keys(d, new_k, old_k):
        d[new_k] = d[old_k]
        del d[old_k]
        return d
    
    all_old_keys = list(state_dict.keys()) # Fixation.
    for k in all_old_keys:
        if 'dil_conv_' in k and 'op.3' in k:
            # for div_conv_ 
            # map the previos op.3 into bn
            new_k = k.replace('op.3', 'bn')
        elif 'sep_conv_' in k:
            # op.1 op.2 is unchanged.
            if 'op.1' in k or 'op.2' in k:
                continue
            elif 'op.3' in k:
                new_k = k.replace('op.3', 'bn1')
            elif 'op.5' in k:
                new_k = k.replace('op.5', 'op2.1')
            elif 'op.6' in k:
                new_k = k.replace('op.6', 'op2.2')
            elif 'op.7' in k:
                new_k = k.replace('op.7', 'bn2')
            else:
                continue
        else:
            continue
        state_dict = _replace_dict_keys(state_dict, new_k, k)
    
    return state_dict

def valid_sort_analysis(root_path,  str_filter='/*/*/*/*/args.json', original_args=False, k='portion'):
    TOP_K = 3
    all_list = glob.glob(root_path + str_filter)
    best_models = {}
    accs = {}
    tr_accs = {}
    kdt = {}
    p_random = {}
    res_dicts = {}
    total_valid_number = 0

    for path in all_list:
        try:
            acc, _kdt, best_model, res_dict, args = basic_statistics_of_given_folder(path, top_k=TOP_K)
        except ValueError as e:
            continue
        space =  args.search_space
        
        target_key = getattr(args, k)
        print(f"Space {space} with {k} {target_key}")
        # loop through the dataset
        # initialize
        for a in [best_models, accs, kdt, p_random, res_dicts]:
            if not space in a.keys():
                a[space] = {}
            if target_key not in a[space].keys():
                a[space][target_key] = []
        
        accs[space][target_key].append(acc)
        kdt[space][target_key].append(_kdt)
        best_models[space][target_key].append(best_model)
        res_dicts[space][target_key].append(res_dict)
        p_random[space][target_key].append(prob_surpass_random(best_model, MAXRANK[space], repeat=TOP_K))
        total_valid_number += 1

    print("Total valid number ", total_valid_number)
    # loaded the performance with random sampling. 
    if original_args:
        return accs, kdt, best_models, p_random, res_dicts
    return accs, kdt, best_models, p_random

