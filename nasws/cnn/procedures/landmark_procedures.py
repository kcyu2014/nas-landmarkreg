"""
Add landmark as regualrization inside.
"""
import torch
import math
from nasws.cnn.procedures.utils_maml import assign_parameters
from utils import accuracy
from .maml_procedure import named_module_parameters, task_update_step
from .maml_procedure import _summarize_shared_train as maml_summarize

import logging
from random import randint

import IPython
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import nasws.cnn.utils
from nasws.cnn.policy.enas_policy import RepeatedDataLoader


"""
In general there are many ways to compute such ranking loss.

- we could directly compare against the loss value
- or we use the distance loss (cosine distance) between data points 

"""

def set_landmark_loss_mode(model, mode: bool):
    if isinstance(model, nn.DataParallel):
        model.module.set_landmark_mode(mode)
    else:
        model.set_landmark_mode(mode)


def _summarize_shared_train(curr_step, loss, rankloss, acc=0, acc_5=0, lr=0.0, epoch_steps=1, writer=None, coef=0):
    """Logs a set of training steps."""
    logging.info(f'| step {curr_step:3d} '
                 f'| lr {lr:.4f} '
                 f'| coef {coef:.4f}'
                 f'| rank loss {rankloss:.2f} '
                 f'| loss {loss:.2f} '
                 f'| acc {acc:8.2f}'
                 f'| acc-5 {acc_5: 8.2f}')



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
    'mae_relu_inverse': lambda l1, l2: F.relu(l2 - l1),
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
    input, target = data
    # for the sake of memory, we do not store the features, but we just compute the graph all the time.
    coeff = args.tmp_landmark_loss_coef
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
        model = change_model_spec_fn(model, landmark_specs[ind])
        curr_loss, _, _ = module_forward_fn(model, input, target, criterion)
        model = change_model_spec_fn(model, landmark_specs[jnd])
        next_loss, _, _ = module_forward_fn(model, input, target, criterion)

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
            rank_obj.update(rank_loss.item(), input.size(0))
        try:
            rank_loss.backward()  # update grad here.
        except Exception as e:
            print(e)
            IPython.embed()


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


def _schedule_coeff(args, lr):
# def _schedule_coeff(args, lr):
    """ Adjust the landscape coefficient """
    if args.landmark_loss_coef_scheduler == 'inverse_lr':
        coeff = (args.learning_rate - lr + args.learning_rate_min) * args.landmark_loss_coef
        logging.info(f"landmark loss coefficient (Inverse lr) {coeff}")
    else:
        coeff = args.landmark_loss_coef

    args.tmp_landmark_loss_coef = coeff
    return args



def adjust_landmark_coef(epoch, args, batch_idx=1, total_batchs=1):
    world_size = args.world_size if hasattr(args, 'world_size') else 1
    if epoch < args.supernet_warmup_epoch:
        coef_adf = 0
    else:
        
        # orig_epoch = epoch
        epoch = epoch - args.supernet_warmup_epoch
        # run_epochs = epoch - args.landmark_warmup_epoch
        if 'increase' in args.landmark_loss_coef_scheduler:
            args.landmark_warmup_epoch = 0
        total_epochs = args.epochs - args.landmark_warmup_epoch - args.supernet_warmup_epoch
        if args.landmark_warmup_epoch > 0 and epoch < args.landmark_warmup_epoch:
            # warming up the landmark coef
            epoch += float(batch_idx + 1) / total_batchs
            if world_size > 1:
                coef_adf = 1. / world_size * (epoch * (world_size - 1) / args.landmark_warmup_epoch + 1)
            else:
                coef_adf = epoch / (args.landmark_warmup_epoch + 1)
        else:
            epoch -= args.landmark_warmup_epoch
            # net epoch 
            if args.landmark_loss_coef_scheduler == "constant":
                coef_adf = 1.
            elif args.landmark_loss_coef_scheduler == "linear_step_decrease": 
                if epoch < total_epochs // 4:
                    coef_adf = 1.
                elif epoch < total_epochs // 2:
                    coef_adf = 1e-1
                elif epoch < total_epochs // 4 * 3:
                    coef_adf = 1e-2
                else:
                    coef_adf = 1e-3
            elif args.landmark_loss_coef_scheduler == "linear_step_increase": 
                if epoch < total_epochs // 4:
                    coef_adf = 0.
                elif epoch < total_epochs // 2:
                    coef_adf = 1e-2
                elif epoch < total_epochs // 4 * 3:
                    coef_adf = 1e-1
                else:
                    coef_adf = 1.
            elif args.landmark_loss_coef_scheduler == "cosine_decrease":
                # self.init_lr * 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
                T_cur = float(epoch * total_batchs) + batch_idx
                T_total = float(total_epochs * total_batchs)
                coef_adf = 0.5  * (1 + math.cos(math.pi * T_cur / T_total))
            elif args.landmark_loss_coef_scheduler == "cosine_increase":
                # self.init_lr * 0.5 * (1 + math.cos(math.pi * T_cur / T_total))
                run_epochs = epoch - args.landmark_warmup_epoch
                total_epochs = args.epochs - args.landmark_warmup_epoch
                T_cur = float(epoch * total_batchs) + batch_idx
                T_total = float(total_epochs * total_batchs)
                # pi to 2 pi, from -1 to 1, i.e. coef adf is from 0 to 1.
                coef_adf = 0.5  * (1 + math.cos(math.pi * (T_cur / T_total + 1)))
            else:
                coef_adf = 0
    coeff = coef_adf * args.landmark_loss_coef
    args.tmp_landmark_loss_coef = coeff
    return coeff



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


def darts_train_model_with_landmark_regularization(
        train_queue, valid_queue, model, criterion, optimizer, lr, args, architect=None,
        search_space=None,
        landmark_loss_step=pairwise_landmark_ranking_loss_step,
        sampler=None,
):
    """
    This is training procedure to add a regularization of ranking loss.

    :param train_queue: Training for the supernet.
    :param valid_queue: Update the ranking loss? or just do nothing.
    :param model:
    :param optimizer:
    :param lr:
    :param args:
    :param architect:
    :param search_space:
    :param landmark_loss:
    :param sampler:
    :return:
    """
    if not isinstance(valid_queue, RepeatedDataLoader):
        valid_queue = RepeatedDataLoader(valid_queue)

    objs = nasws.cnn.utils.AverageMeter()
    top1 = nasws.cnn.utils.AverageMeter()
    top5 = nasws.cnn.utils.AverageMeter()
    rank_objs = nasws.cnn.utils.AverageMeter()
    coeff = adjust_landmark_coef(args.current_epoch, args)

    for step, (input, target) in enumerate(train_queue):
        model.train()
        if args.debug and step > 10:
            logging.warning('Testing only. Break after 10 batches.')
            break

        if sampler:
            model = sampler(model, architect, args)

        n = input.size(0)
        input = input.cuda().requires_grad_()
        target = target.cuda()

        if architect and args.current_epoch >= args.epochs:
            # after warmup
            search_input, search_target = valid_queue.next_batch()
            search_input = search_input.cuda()
            search_target = search_target.cuda(non_blocking=True)
            architect.step(input, target, search_input, search_target, lr, optimizer, unrolled=args.policy_args.unrolled)

        # Update model
        optimizer.zero_grad()
        # update the normal parameters.
        loss, logits, _ = search_space.module_forward_fn(model, input, target, criterion)
        loss.backward()

        if args.current_epoch >= args.supernet_warmup_epoch: 
            # add ranking loss backwards.
            if args.landmark_use_valid:
                rank_input, rank_target = valid_queue.next_batch()
                rank_input = rank_input.cuda().requires_grad_()
                rank_target = rank_target.cuda()
            else:
                rank_input, rank_target = input, target
            # proceed ranking steps.
            # logging.debug('compute the rank loss!')
            set_landmark_loss_mode(model, True)
            landmark_loss_step(model, (rank_input, rank_target), search_space, criterion, args,
                            search_space.change_model_spec, search_space.module_forward_fn,
                            rank_objs)
            set_landmark_loss_mode(model, False)
        
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = nasws.cnn.utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            _summarize_shared_train(step, objs.avg, rank_objs.avg, top1.avg, top5.avg, lr, coef=coeff)
    
    # report ranking loss here.
    logging.info(f"Loss at epoch end -> {objs.avg + rank_objs.avg} = (rank) {rank_objs.avg} + (model) {objs.avg}")
    return top1.avg, objs.avg


def maml_ranking_loss_procedure(meta_queue, task_queue, model, criterion, optimizer, lr, args,
                                search_space,
                                landmark_loss_step=pairwise_landmark_ranking_loss_step,
                                sampler=None, architect=None,
                                ):
    """
    Combine MAML with this.

    Checked the memory, is correctly reset.
    :param model:
    :param tasks:
    :param meta_queue: data for updating the SuperNet, i.e. meta network, this should be normal data-loader.
    :param task_queue: data for computing "task" gradient, this should be RepeatedDataloader
    :param args:
    :param valid_queue:
    :param sampler: sample the architecture.
    :return:
    """
    # pass for now, disable.
    objs = nasws.cnn.utils.AverageMeter()
    top1 = nasws.cnn.utils.AverageMeter()
    top5 = nasws.cnn.utils.AverageMeter()
    task_objs = nasws.cnn.utils.AverageMeter()
    task_top1 = nasws.cnn.utils.AverageMeter()
    task_top5 = nasws.cnn.utils.AverageMeter()
    rank_objs = nasws.cnn.utils.AverageMeter()
    _schedule_coeff(args, lr)
    if not isinstance(task_queue, RepeatedDataLoader):
        task_queue = RepeatedDataLoader(task_queue)

    num_inner_tasks = args.maml_num_inner_tasks
    task_lr = args.maml_task_lr if args.maml_task_lr > 0 else lr
    # logging.debug("Task lr is {}".format(task_lr))

    num_episodes = len(meta_queue) // num_inner_tasks
    meta_parameters = named_module_parameters(model)

    logging.info(f"Epoch episode = {num_episodes}")
    meta_queue = iter(meta_queue)
    for episode in range(num_episodes):
        if args.debug and episode > 2:
            logging.warning('Testing only, break after 2 episodes.')
            break

        meta_loss = 0.0
        total_model = 0
        optimizer.zero_grad()
        # 880Mb before this start
        # Aggregate over num_inner_tasks sub-graph.
        for n_task in range(num_inner_tasks):
            input, target = task_queue.next_batch()
            input = input.cuda().requires_grad_()
            target = target.cuda()
            # IPython.embed(helper='checking the debugging')
            if sampler:
                model = sampler(model, architect, args)
            # IPython.embed(header='Checking a_dict memory using or not ')
            # compute one step for task udpate.
            # assign the meta-parameter back.
            a_dict = task_update_step(model, (input, target), task_lr, criterion, args,
                                      task_objs, task_top1, task_top5)

            # Compute gradient for meta-network
            meta_input, meta_target = next(meta_queue)
            n = meta_input.size(0)
            meta_input = meta_input.cuda().requires_grad_()
            meta_target = meta_target.cuda()

            # compute the meta_loss and do grad backward
            # assign_parameters(model, a_dict)
            meta_logits, meta_aux_logits = model(meta_input)
            loss = criterion(meta_logits, meta_target)
            if meta_aux_logits is not None:
                meta_aux_loss = criterion(meta_aux_logits, meta_target)
                loss += 0.4 * meta_aux_loss
            loss.backward()

            # add ranking loss backwards.
            rank_input, rank_target = meta_input, meta_target
            landmark_ids, landmark_specs = search_space.landmark_topologies

            # proceed ranking steps.
            landmark_loss_step(model, (rank_input, rank_target), landmark_ids, landmark_specs, criterion, args,
                               search_space.change_model_spec, search_space.module_forward_fn,
                               rank_objs)

            # IPython.embed(header='check meta_parameter has grad or not.')
            meta_loss = meta_loss + loss.item()
            # keep computing the backward gradients throught the time.
            total_model += 1
            objs.update(loss.item(), n)
            prec1, prec5 = accuracy(meta_logits.detach(), meta_target, topk=(1,5))
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            assign_parameters(model, meta_parameters)
            del a_dict
            # 1400M at first iter, increase 2M every epoch. but it is reasonable,
            # because keep adding gradients of meta_parameters.

        if episode % args.report_freq == 0:
            maml_summarize(episode, objs.avg, task_objs.avg, top1.avg, top5.avg,
                                    task_lr, task_top1.avg, task_top5.avg)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        # conclude one MAML update loop.
        # IPython.embed(header='Checking memory is released for next episode loop')
        # Checked with the memory, after entire loop, it is correctly cleaned the pointers.
    logging.info(f"Loss at epoch end -> {objs.avg + rank_objs.avg} = (rank) {rank_objs.avg} + (model) {objs.avg}")
    return (top1.avg, task_top1.avg), (objs.avg, task_objs.avg)


