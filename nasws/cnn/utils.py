"""
Collection of CNN utils, that are duplicated by others.

"""

from collections import namedtuple

import torch
import numpy as np
from scipy.stats import kendalltau, spearmanr

Rank = namedtuple('Rank', 'valid_acc valid_obj geno_id gt_rank')


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def clip_grad_norm(grad_tensors, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.
    Modify from the original ones, just to clip grad directly.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        grad_tensors (Iterable[Tensor] or Tensor): an iterable of grad Tensors
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(grad_tensors, torch.Tensor):
        grad_tensors = [grad_tensors]
    grad_tensors = list(filter(lambda p: p is not None, grad_tensors))
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == 'inf':
        total_norm = max(p.data.abs().max() for p in grad_tensors)
    else:
        total_norm = 0
        for p in grad_tensors:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in grad_tensors:
            p.data.mul_(clip_coef)
    return total_norm


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res



### rank related 

# sparse ranking things
def sparse_rank_by_simple_bin(ranks, perfs, threshold=1e-4, verbose=False):
    """
    Simple reduce the ranking by giving bin by threshold
    for example,
        threshold=1e=2, [0.911, 0.9112, 0.912] -> [1,1,2]
        threshold=1e=3, [0.911, 0.9112, 0.912] -> [1,2,3]

    :param ranks:
    :param perfs:
    :param threshold:
    :return:
    """
    new_ranks = []
    c_perf = 0.0
    c_rank = -1

    if threshold < 1.:
        multipler = 1 / threshold
    else:
        multipler = 1.
    rank_map = {}
    for r, p in zip(ranks, perfs):
        p_rank = p * multipler // 1
        cp_rank = c_perf * multipler // 1
        if p_rank > cp_rank:
            c_rank += 1
        rank_map[r] = c_rank
        new_ranks.append(c_rank)
        c_perf = p
        if verbose:
            print(f'map ({p}) from {r} to {c_rank}')

    return new_ranks, rank_map


def sort_hash_perfs(hashs, perfs, verbose=False):
    """Sort the hash based on perfs. 
        
    Parameters
    ----------
    hashs : list
        list as id
    perfs : list
        list to be sorted with.
    verbose : bool, optional
        verbose, by default False
    
    Returns
    -------
    list, list:
        sorted_hashs, sorted_perfs.
    """
    # make sure this is duplicated.
    import copy
    hashs = copy.deepcopy(hashs)
    perfs = copy.deepcopy(perfs)
    sorted_indices = np.argsort(perfs)
    if verbose:
        print(sorted_indices)
    s_h, s_p = [hashs[i] for i in sorted_indices], [perfs[i] for i in sorted_indices]
    return s_h, s_p


def compute_sparse_rank_metric(model_ids, model_perfs, gt_perfs, threshold=1e-4,
                              fn_map_perf_to_new_rank=sparse_rank_by_simple_bin,
                              metric_fn=kendalltau,
                              verbose=False):
    """
    Compute the sparse kendall tau, by compression
    :param model_ids: [gtid_1, gtid_2, ...]
    :param model_perfs: [perf_1, perf_2, ...]
    :param gt_perfs: [gt_perf_1, ...]
    :param threshold:
    :param fn_map_perf_to_new_rank:
    :return:
    """
    # wrap the perf and gt_perfs into [0, 1]
    avg_perfs_multiplier = 1e-2 if 1 < np.average(model_perfs) < 100 else 1.
    model_perfs = [p * avg_perfs_multiplier for p in model_perfs]
    avg_perfs_multiplier = 1e-2 if 1 < np.average(gt_perfs) < 100 else 1.
    gt_perfs = [p * avg_perfs_multiplier for p in gt_perfs]
    # s_model_perfs, s_gt_perfs = sort_hash_perfs(model_perfs, gt_perfs, verbose)

    s_model_ids, s_model_perfs = sort_hash_perfs(model_ids, model_perfs, verbose=verbose)
    gt_model_ids, s_gt_perfs = sort_hash_perfs(model_ids, gt_perfs, verbose=verbose)
    if verbose:
        print("sorted model ids by pred-perf", s_model_ids)
    sgt_sparse_ranks, sgt_rank_map = fn_map_perf_to_new_rank(
        gt_model_ids, s_gt_perfs, threshold=threshold, verbose=verbose)
    pred_sparse_ranks = [sgt_rank_map[i] for i in s_model_ids]
    if verbose:
        print(pred_sparse_ranks)
    return metric_fn(sgt_sparse_ranks, pred_sparse_ranks)


def compute_sparse_rank_metric_old(model_ids, model_perfs, gt_perfs, threshold=1e-4,
                              fn_map_perf_to_new_rank=sparse_rank_by_simple_bin,
                              metric_fn=kendalltau):
    """
    Compute the sparse kendall tau, by compression
    :param model_ids: [gtid_1, gtid_2, ...]
    :param model_perfs: [perf_1, perf_2, ...]
    :param gt_perfs: [gt_perf_1, ...]
    :param threshold:
    :param fn_map_perf_to_new_rank:
    :return:
    """
    avg_perfs_multiplier = 1e-2 if 1 < np.average(model_perfs) < 100 else 1.
    model_perfs = [p * avg_perfs_multiplier for p in model_perfs]
    print(model_perfs)
    gt_perfs, _ = sort_hash_perfs(gt_perfs, model_perfs)
    print(gt_perfs)
    sgt_model_ids, sgt_perfs = sort_hash_perfs(model_ids, gt_perfs, verbose=False)
    print(sgt_model_ids)
    sgt_sparse_ranks = fn_map_perf_to_new_rank(sgt_model_ids, sgt_perfs, threshold=threshold)
    print(sgt_sparse_ranks)
    pred_sparse_ranks = [sgt_sparse_ranks[sgt_model_ids.index(i)] for i in model_ids]
    print(pred_sparse_ranks)
    print("Reduced ranks from {} to {}".format(len(set(model_ids)), len(set(sgt_sparse_ranks))))
    return metric_fn(sgt_sparse_ranks, pred_sparse_ranks)


def compute_sparse_kendalltau(model_ids, model_perfs, gt_perfs, threshold=1e-4,
                              fn_map_perf_to_new_rank=sparse_rank_by_simple_bin, verbose=False):
        
    return compute_sparse_rank_metric(model_ids, model_perfs, gt_perfs, threshold=threshold,
                              fn_map_perf_to_new_rank=fn_map_perf_to_new_rank,
                              metric_fn=kendalltau,
                              verbose=verbose)

def compute_sparse_spearmanr(model_ids, model_perfs, gt_perfs, threshold=1e-4,
                              fn_map_perf_to_new_rank=sparse_rank_by_simple_bin):
        
    return compute_sparse_rank_metric(model_ids, model_perfs, gt_perfs, threshold=threshold,
                              fn_map_perf_to_new_rank=fn_map_perf_to_new_rank,
                              metric_fn=spearmanr)


def compute_percentile(model_ranks, num_total_arch, top_k=10, verbose=False):
    """
    Compute and print the percentile of a nasbench top models
    :param model_ranks: descending order, [0] is the best
    :param num_total_arch: Total number of archs to compute the percentile
    :param top_k: to reduce model ranks.
    :return:
    """
    # IPython.embed()
    r = np.asanyarray(model_ranks[:top_k], dtype=np.float)
    percentile = r / num_total_arch
    if verbose:
        print("Percentile of top {}: {} - {} - {}".format(top_k, percentile.min(), np.median(percentile), percentile.max()) )
        # should not compute the mean since percentile is more like categorical data.
    return percentile


def rectify_model_ids_byvalid_to_bytest(model_ids, nasbench, hash_perfs):
    """
    Map from the model_id ranked by validation to ranked by testing.

    :param model_ids:
    :param nasbench:
    :param hash_perfs:
    :return:
    """
    # add logic if hash_perf is none

    if hash_perfs is None:
        # raise NotImplementedError("Not yet supported. add this to nasbench later.")
        hash_perfs = {}
        for h, p in zip(nasbench.hash_rank, nasbench.perf_rank):
            hash_perfs[h] = p

    hashs = nasbench.hash_rank
    perfs = [hash_perfs[h] for h in hashs]
    hashs_bytest, perfs_bytest = sort_hash_perfs(hashs, [p[1] for p in perfs])
    hashs_byvalid, _ = sort_hash_perfs(hashs, [p[0] for p in perfs])
    rect_model_ids = [hashs_bytest.index(hashs_byvalid[i]) for i in model_ids]
    return rect_model_ids, [perfs_bytest[i] for i in rect_model_ids]


# function begins