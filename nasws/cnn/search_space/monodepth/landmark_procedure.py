"""
Add landmark as regualrization inside.
"""
import torch
import math


import logging
from random import randint

import IPython
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


"""
In general there are many ways to compute such ranking loss.

- we could directly compare against the loss value
- or we use the distance loss (cosine distance) between data points 

"""

class SearchSpace:

    def __init__(self, ids, specs, weights):
        super().__init__()
        self._landmark_ids = ids
        self._landmark_specs = specs
        self._landmark_weights = weights

    @property
    def landmark_topologies(self):
        return self._landmark_ids, self._landmark_specs

    @property
    def landmark_weights(self):
        return self._landmark_weights
    

def _summarize_shared_train(curr_step, loss, rankloss, acc=0, acc_5=0, lr=0.0, epoch_steps=1, writer=None):
    """Logs a set of training steps."""
    logging.info(f'| step {curr_step:3d} '
                 f'| lr {lr:4.2f} '
                 f'| rank loss {rankloss:.2f} '
                 f'| loss {loss:.2f} '
                 f'| acc {acc:8.2f}'
                 f'| acc-5 {acc_5: 8.2f}')


def adjust_landmark_coef(epoch, args, batch_idx=1, total_batchs=1):
    world_size = args.world_size
    if args.landmark_warmup_epoch > 0 and epoch < args.landmark_warmup_epoch:
        # warming up the landmark coef
        epoch += float(batch_idx + 1) / total_batchs
        if world_size > 1:
            coef_adf = 1. / world_size * (epoch * (world_size - 1) / args.landmark_warmup_epoch + 1)
        else:
            coef_adf = min(1, epoch / (args.landmark_warmup_epoch + 1))
    elif args.landmark_loss_coef_scheduler == "constant":
        coef_adf = 1.
    elif args.landmark_loss_coef_scheduler == "linear_step_decrease": 
        if epoch < 100:
            coef_adf = 1.
        elif epoch < 150:
            coef_adf = 1e-1
        elif epoch < 200:
            coef_adf = 1e-2
        else:
            coef_adf = 1e-3
    elif args.landmark_loss_coef_scheduler == "linear_step_increase": 
        if epoch < 100:
            coef_adf = 0.
        elif epoch < 150:
            coef_adf = 1e-2
        elif epoch < 200:
            coef_adf = 1e-1
        else:
            coef_adf = 1.
    elif args.landmark_loss_coef_scheduler == "cosine_decrease":
        # self.init_lr * 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
        run_epochs = epoch - args.landmark_warmup_epoch
        total_epochs = args.epochs - args.landmark_warmup_epoch
        T_cur = float(run_epochs * total_batchs) + batch_idx
        T_total = float(total_epochs * total_batchs)
        coef_adf = 0.5  * (1 + math.cos(math.pi * T_cur / T_total))

    return coef_adf * args.landmark_loss_coef


def rank_cross_entropy_loss(l1, l2):
    logit = torch.sigmoid(l1-l2)
    # logit = F.prelu(logit, torch.tensor(0.1))
    return -torch.log(1 - logit)

def rank_infinite_loss_v1(l1, l2, w1, w2):
    d = (l1 - l2) / (w1 - w2) 
    # in this case: l1 < l2 and w1 < w2, or l1 > l2 and w1 > w2, d > 0. but we should 
    p = torch.sigmoid(-d)
    return F.relu(0.5 - p)


def rank_infinite_loss_v2(l1, l2, w1, w2):
    d = (l1 - l2) / (w1 - w2)
    p = torch.sigmoid(-d)
    return F.softplus(0.5 - p)


def rank_infinite_relu(l1, l2, w1, w2):
    d = (l1 - l2) * (w1 - w2)
    return F.relu(d)


def rank_infinite_softplus(l1, l2, w1, w2):
    d = (l1 - l2) * (w1 - w2)
    return F.softplus(d, beta=5)


def rank_hinge_sign_infinite(l1, l2, w1, w2):
    return F.relu(1 - (l1 - l2) * torch.sign(w2 - w1))


def rank_cross_entropy_focal_loss(l1, l2, gamma=5):
    logit = torch.sigmoid(l1 - l2)
    # logit = F.prelu(logit, torch.tensor(0.1))
    return - (logit).pow(gamma) * torch.log(1 - logit)


def rank_mixed_cross_entropy_loss(l1, l2):
    if l1 < l2:
        return rank_cross_entropy_focal_loss(l1, l2)
    else:
        return rank_cross_entropy_loss(l1, l2)


def tanh_sign_infinite(l1, l2, w1, w2):
    # given the fact that, l1 < l2 == w1 > w2. 
    l = torch.tanh(l1 - l2) * torch.sign(w1 - w2)
    return F.relu(l)


def tanh_infinite(l1, l2, w1, w2):
    # given the fact that, l1 < l2 == w1 > w2. 
    l = torch.tanh(l1 - l2) * torch.tanh(w1 - w2)
    return F.relu(l)


def tanh_infinite_norelu(l1, l2, w1, w2):
    # given the fact that, l1 < l2 == w1 > w2. 
    return torch.tanh(l1 - l2) * torch.tanh(w1 - w2)
    

# to compute the rank loss for each pair of input losses.
_loss_fn = {
    'mae_relu': lambda l1, l2 : F.relu(l1 - l2),
    'mae_sign_relu': lambda l1, l2 : F.relu(torch.sign(l1 - l2)),
    'mae_sign_tanh_relu': lambda l1, l2: F.relu(torch.sign(torch.tanh(l1 - l2))),
    'mae_tanh_relu': lambda l1, l2: F.relu(torch.tanh(l1 - l2)),
    'mae_softplus': lambda l1, l2: F.softplus(l1 - l2),
    'mae_softplus_beta3': lambda l1, l2: F.softplus(l1 - l2, beta=3),
    'mae_softplus_beta5': lambda l1, l2: F.softplus(l1 - l2, beta=5),
    'mae_softplus_beta7': lambda l1, l2: F.softplus(l1 - l2, beta=7),
    'focal_loss': rank_cross_entropy_focal_loss,
    'mae_relu_norm': lambda l1, l2 : F.relu((l1 - l2) / (l1 - l2).abs() * (l1 + l2) / 2),
    'mae_tanh_infinite': tanh_infinite,
    'tanh_infinite': tanh_infinite_norelu,
    'mae_sign_tanh_infinite': tanh_sign_infinite,
    'mae_relu_sigmoid_infinite': rank_infinite_loss_v1,
    'mae_relu_infinite': rank_infinite_relu,
    'softplus_infinite': rank_infinite_softplus,
    'sigmoid_softplus_infinite': rank_infinite_loss_v2,
    'hinge_sign_infinite': rank_hinge_sign_infinite,
    'crossentropy': rank_cross_entropy_loss,
    'mixed_focal': rank_mixed_cross_entropy_loss,
}


def get_rank_loss_fn(name, weighted):
    """
    All of these loss will penalize l1 > l2, i.e. ground truth label is l1 < l2.

    :param name: args.landmark_loss_fn 
    :param weighted: weighted to add a subscript.
    :return: loss fn.
    """

    if weighted == 'embed':
        return lambda l1, l2, w : w * _loss_fn[name](l1, l2)
    elif weighted == 'infinite':
        return _loss_fn[name + '_infinite']
    return _loss_fn[name]


def pairwise_landmark_ranking_loss_step(model, data, search_space, criterion, args,
                                        change_model_spec_fn, module_forward_fn,
                                        rank_obj=None, pair_indicies=None):
    """
    Compute the ranking loss:
        for landmark models, m1, m2, ...., suppose they are in descending order
        FOR i > j,  L_rank = sum_{i,j} max(0, L(m_j) - L(m_i))

        if landmark_loss_adjacent, i = j + 1 for sure. otherwise, i = j+1, j+2 ..., n

    Version 1.0

    :param model:
    :param data:
    :param search_space:
    :param criterion:
    :param args:
    :param change_model_spec_fn: change model spec function, this should be associated with search space.
    :param module_forward_fn: return the current loss and next loss for now.
    :param rank_obj:
    :param landmark_weights: weights you would like to associate with each landmark architecture.
    :return:
    """
    # input, target = data
    # for the sake of memory, we do not store the features, but we just compute the graph all the time.
    coeff = adjust_landmark_coef(args.tmp_epoch, args)
    rank_loss_fn = get_rank_loss_fn(args.landmark_loss_fn, args.landmark_loss_weighted)
    landmark_ids, landmark_specs = search_space.landmark_topologies
    if pair_indicies is None:
        pair_indicies = []
        for ind, model_id in enumerate(landmark_ids[:-1]):  # skip the last one
            end = min(ind + 1 + args.landmark_loss_adjacent_step, len(landmark_ids)) \
                    if args.landmark_loss_adjacent else len(landmark_ids)
            for jnd in range(ind+1, end):
                pair_indicies.append((ind, jnd))
    
    for ind, jnd in pair_indicies:
        # currently, landmarks id loss should decrease!
        # change the model to current one
        change_model_spec_fn(model, landmark_specs[ind])
        curr_loss, _, _ = module_forward_fn(model, data, criterion)
        change_model_spec_fn(model, landmark_specs[jnd])
        next_loss, _, _ = module_forward_fn(model, data, criterion)
        # IPython.embed()
        # weighted landmark
        if args.landmark_loss_weighted == 'embed':
            landmark_weights = search_space.landmark_weights
            rank_loss = coeff * rank_loss_fn(next_loss, curr_loss, abs(landmark_weights[ind] - landmark_weights[jnd]))
        elif args.landmark_loss_weighted == 'infinite':
            landmark_weights = search_space.landmark_weights
            rank_loss = coeff * rank_loss_fn(
                next_loss, curr_loss, 
                torch.tensor(landmark_weights[jnd]).float(), 
                torch.tensor(landmark_weights[ind]).float(),
                )
        else:
            rank_loss = coeff * rank_loss_fn(next_loss, curr_loss)
        
        if rank_obj:
            rank_obj.update(rank_loss.item(), data['image'].size(0))
            rank_obj.landmark_coef = coeff # track the coefficient over epoch
        try:
            rank_loss.backward()  # update grad here.
        except Exception as e:
            print(e)
            IPython.embed()
    
    return rank_loss.item()

def random_pairwise_loss_step(model, data, search_space, criterion, args,
                              change_model_spec_fn, module_forward_fn,
                              rank_obj=None):
    # each time, random a pair and compare.
    pairwise_indicies = []
    # landmark_ids, landmark_specs = search_space.landmark_topologies
    # landmark_weights = search_space.landmark_weights
    num_landmark = len(search_space._landmark_ids)
    for _ in range(args.landmark_loss_random_pairs):
        a = randint(0, num_landmark - 2)
        b = randint(a+1, num_landmark - 1)
        pairwise_indicies.append([a, b])
    return pairwise_landmark_ranking_loss_step(model, data, search_space, criterion, args,
                                               change_model_spec_fn, module_forward_fn, rank_obj,
                                               pairwise_indicies)


def random_three_pairwise_loss_step(model, data, search_space, criterion, args,
                                    change_model_spec_fn, module_forward_fn,
                                    rank_obj=None):
    # each time, random 3 architecture to formulate this
    pairwise_indicies = []
    num_landmark = len(search_space._landmark_ids)
    for _ in range(args.landmark_loss_random_pairs):
        a, b, c = sorted(np.random.choice(np.arange(num_landmark), 3, replace=False).tolist())
        pairwise_indicies.append([a, b])
        pairwise_indicies.append([b, c])
    # IPython.embed(header='check three pair')
    # here specs contains landmark weights
    return pairwise_landmark_ranking_loss_step(model, data, search_space, criterion, args,
                                               change_model_spec_fn, module_forward_fn, rank_obj,
                                               pairwise_indicies)


# def _schedule_coeff(args, lr):
#     if args.landmark_loss_coef_scheduler == 'inverse_lr':
#         coeff = (args.learning_rate - lr + args.learning_rate_min) * args.landmark_loss_coef
#         logging.info(f"landmark loss coefficient (Inverse lr) {coeff}")
#     else:
#         coeff = args.landmark_loss_coef

#     args.tmp_landmark_loss_coef = coeff
#     return args


landmark_loss_step_fns = {
    'pairwise_loss': pairwise_landmark_ranking_loss_step,
    # 'pairwise_loss_normalize': pairwise_landmark_ranking_loss_step,
    'random_pairwise_loss': random_pairwise_loss_step,
    # 'random_pairwise_infinite_loss_v1': random_pairwise_loss_step,
    'random_pariwise_loss_cross_entropy': random_pairwise_loss_step,
    'random_pairwise_loss_mixed_focal': random_pairwise_loss_step,
    'random_three_pairwise_loss': random_three_pairwise_loss_step,
    'pairwise_logits': NotImplementedError("not yet implemented.")
}
