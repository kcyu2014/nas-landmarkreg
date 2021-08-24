import gc
import logging
import operator
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

# from nasws.cnn.cnn_general_search_policies import CNNSearchPolicy
from nasws.cnn.policy.enas_policy import RepeatedDataLoader
from nasws.cnn.procedures import darts_train_model
from nasws.cnn.search_space.search_space import CNNSearchSpace
from nasws.cnn.utils import Rank, AverageMeter, accuracy
from utils import clone_detach_model_state_dict
from visualization.process_data import tensorboard_summarize_list


def evaluate_normal(self, model, fitnesses_dict, model_spec_id_pool, eval_queue, change_model_fn, criterion):
    """
    Evaluate procedure. Current for NASBench, maybe adapted for others.

    :param self:
    :param model: model to be trained
    :param fitnesses_dict: Not sure if we still need this.
    :param model_spec_id_pool: evaluate architecture ids. Pass into self.search_space.topologies[_id]
    :param eval_queue:
    :param change_model_fn: change model by topologies
    :param criterion: loss to use
    :return: list
        1. rank_gens: {model_spec_hash: Rank[acc, obj, id, GT-rank]}
            Ranking results

        2. eval_result: {model_spec_id: (acc, obj)}
            the evaluation results are also returned.
    """
    logging.info("Evaluating {} architectures".format(len(model_spec_id_pool)))
    fitnesses_dict = fitnesses_dict or {}
    total_avg_acc = 0
    total_avg_obj = 0
    # rank dict for the possible solutions
    model_specs_rank = {}
    # as backup
    ind = 0
    eval_result = {}
    while ind < len(model_spec_id_pool):
        # model spec id to test.
        # computing the genotype of the next particle
        # recover the weights
        if self.args.debug:
            if ind > 10:
                logging.debug("Break after evaluating 10 architectures. Total {}".format(len(model_spec_id_pool)))
                break
        
        model_spec_id = model_spec_id_pool[ind]
        ind += 1  # increment this.
        
        if isinstance(model_spec_id, int):
            new_model_spec = self.search_space.topologies[model_spec_id]
        elif isinstance(model_spec_id, str):
            model_spec_id = self.search_space.hashs.index(model_spec_id)
            new_model_spec = self.search_space.topologies[model_spec_id]
        else:
            new_model_spec = model_spec_id
            model_spec_id = self.search_space.topology_to_id(new_model_spec)
            
            if model_spec_id is None:
                # serialize this instead of using the result directly
                model_spec_id = self.search_space.serialize_model_spec(new_model_spec)
                logging.warning(f'{new_model_spec} not in the search space, ignore in ranking computation.')

        # selecting the current subDAG in our DAG to train
        model = change_model_fn(model, new_model_spec)
        # Reset the weights.
        # evaluate before train
        logging.info('evaluate the model spec id: {}'.format(model_spec_id))
        _avg_val_acc, _avg_val_obj = self.eval_fn(eval_queue, model,
                                                  criterion=criterion, verbose=False)
        eval_result[model_spec_id] = _avg_val_acc, _avg_val_obj
        # update the total loss.
        total_avg_acc += _avg_val_acc
        total_avg_obj += _avg_val_obj

        # saving the particle fit in our dictionaries
        fitnesses_dict[model_spec_id] = _avg_val_acc

        if isinstance(model_spec_id, int):
            ms_hash = self.search_space.hashs[model_spec_id]
            model_specs_rank[ms_hash] = Rank(_avg_val_acc, _avg_val_obj, model_spec_id,
                                            self.search_space.rank_by_mid[model_spec_id])
        
        # else, just do not generate this model_spec rank, only keep the eval result.
        # manual collect the non-used graphs.
        gc.collect()

    # save the ranking, according to their GENOTYPE but not particle id
    rank_gens = sorted(model_specs_rank.items(), key=operator.itemgetter(1))
    # hash to positions mapping, before training
    return rank_gens, eval_result


def evaluate_extra_steps(policy, epoch, data_source, fitnesses_dict=None, train_queue=None, eval_pool=None):
    """
    Full evaluation of all possible models, after training each model for a fixed batch.

    :param epoch:
    :param data_source:
    :param fitnesses_dict: Store the model_spec_id -> accuracy
    :return:
    """
    nb = policy.args.neweval_num_train_batches
    assert nb > 0
    if not train_queue:
        raise ValueError("New evaluation scheme requires a training queue.")

    fitnesses_dict = fitnesses_dict or {}
    top1 = AverageMeter()
    objs = AverageMeter()

    # rank dict for the possible solutions
    model_specs_rank = {}
    model_specs_rank_before = {}

    # make sure the backup weights on CPU, do not occupy the space
    model = policy.parallel_model
    backup_weights = clone_detach_model_state_dict(model)
    # backup_weights = model.cpu().state_dict()
    # model.cuda()

    _train_iter = enumerate(train_queue)  # manual iterate the data here.
    _train_queue = []

    ind = 0
    eval_before_train = {}
    eval_after_train = {}
    eval_pool = eval_pool or policy.search_space.evaluate_model_spec_id_pool()

    while ind < len(eval_pool):
        # model spec id to test.
        # computing the genotype of the next particle
        # recover the weights
        try:
            if policy.args.debug:
                if ind > 10:
                    logging.debug("Break after evaluating 10 architectures. Total {}".format(len(eval_pool)))
                    break
            model_spec_id = eval_pool[ind]
            ind += 1  # increment this.
            new_model_spec = policy.search_space.topologies[model_spec_id]

            # selecting the current subDAG in our DAG to train
            model = policy.search_space.change_model_spec(model, new_model_spec)

            # Reset the weights.
            logging.debug('Resetting parallel model weights ...')
            model.load_state_dict(backup_weights)
            model.cuda()  # make sure this is on GPU.

            # evaluate before train
            _avg_val_acc_before, _avg_val_obj_before = policy.eval_fn(data_source, policy.parallel_model,
                                                                      criterion=policy.criterion, verbose=False)

            eval_before_train[model_spec_id] = _avg_val_acc_before, _avg_val_obj_before

            # IPython.embed(header='check if this is always the same for everyone')
            _train_queue = policy.next_batches(train_queue, nb)
            _batch_count = len(_train_queue)

            # IPython.embed(header=f'Model {model_spec_id}: finish before eval, == checked')
            logging.debug('Train {} batches for model_id {} before eval'.format(_batch_count, model_spec_id))
            lr = policy.scheduler.get_lr()[0]
            org_train_acc, org_train_obj = policy.eval_fn(_train_queue, policy.parallel_model,
                                                          criterion=policy._loss, verbose=policy.args.debug)
            # too complicated, need to sort this out.
            train_acc, train_obj = policy.eval_train_fn(
                _train_queue, None, policy.parallel_model, policy._loss, policy.optimizer, lr)

            # clean up the train queue completely.
            for d in _train_queue:
                del d
            _train_queue = []  # clean the data, destroy the graph.

            logging.debug('-> Train acc {} -> {} | train obj {} -> {} '.format(
                org_train_acc, train_acc, org_train_obj, train_obj))

            logging.info('evaluate the model spec id: {}'.format(model_spec_id))
            _avg_val_acc, _avg_val_obj = policy.eval_fn(data_source, policy.parallel_model,
                                                        criterion=policy._loss, verbose=False)
            eval_after_train[model_spec_id] = _avg_val_acc, _avg_val_obj
            logging.info('eval acc {} -> {} | eval obj {} -> {}'.format(
                _avg_val_acc_before, _avg_val_acc, _avg_val_obj_before, _avg_val_obj
            ))
            top1.update(_avg_val_acc, 1)
            objs.update(_avg_val_obj, 1)

            # saving the particle fit in our dictionaries
            fitnesses_dict[model_spec_id] = _avg_val_acc
            ms_hash = policy.search_space.hashs[model_spec_id]
            model_specs_rank[ms_hash] = Rank(_avg_val_acc, _avg_val_obj, model_spec_id,
                                             policy.search_space.rank_by_mid[model_spec_id])
            model_specs_rank_before[ms_hash] = Rank(_avg_val_acc_before, _avg_val_obj_before, model_spec_id,
                                                    policy.search_space.rank_by_mid[model_spec_id])
            gc.collect()

        except StopIteration as e:
            _train_iter = enumerate(train_queue)
            logging.debug("Run out of train queue, {}, restart ind {}".format(e, ind - 1))
            ind = ind - 1

    rank_gens = sorted(model_specs_rank.items(), key=operator.itemgetter(1))
    rank_gens_before = sorted(model_specs_rank_before.items(), key=operator.itemgetter(1))
    rank_gens_before_pos = {elem[0]: pos for pos, elem in enumerate(rank_gens_before)}

    # update the ranking.
    if 'ranking_per_epoch' not in policy.running_stats.keys():
        policy.running_stats['ranking_per_epoch'] = OrderedDict()
    policy.running_stats['ranking_per_epoch'][epoch] = rank_gens

    if 'ranking_per_epoch_before' not in policy.running_stats.keys():
        policy.running_stats['ranking_per_epoch_before'] = OrderedDict()
    policy.running_stats['ranking_per_epoch_before'][epoch] = rank_gens_before
    policy.eval_result[epoch] = (eval_before_train, eval_after_train)

    policy.logger.info('VALIDATION RANKING OF PARTICLES')
    for pos, elem in enumerate(rank_gens):
        policy.logger.info(f'particle gen id: {elem[1].geno_id}, acc: {elem[1].valid_acc}, obj {elem[1].valid_obj}, '
                           f'hash: {elem[0]}, pos {pos} vs orig pos {rank_gens_before_pos[elem[0]]}')

    if policy.writer:
        # process data into list.
        accs_before, objs_before = zip(*eval_before_train.values())
        accs_after, objs_after = zip(*eval_after_train.values())
        tensorboard_summarize_list(accs_before, writer=policy.writer, key='neweval_before/acc', step=epoch,
                                   ascending=False)
        tensorboard_summarize_list(accs_after, writer=policy.writer, key='neweval_after/acc', step=epoch,
                                   ascending=False)
        tensorboard_summarize_list(objs_before, writer=policy.writer, key='neweval_before/obj', step=epoch)
        tensorboard_summarize_list(objs_after, writer=policy.writer, key='neweval_after/obj', step=epoch)

    return fitnesses_dict


def enas_validation_full_model(valid_queue, model, dag, criterion, args, verbose=True):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            if args.debug:
                if step > 10:
                    print("Break after 10 batch")
                    break
            input = input.cuda()
            target = target.cuda()
            # target = target.cuda(async=True)
            logits, _ = model(input, dag)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0 and step > 0 and verbose:
                logging.info('valid | step %03d | loss %e | acc %f | acc-5 %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def evaluate_enas_with_dag(model, controller, valid_queue, args, return_perf):
    total_loss = AverageMeter()
    total_top1 = AverageMeter()
    model.eval()
    controller.eval()
    archs_pool = []
    archs_perf = []

    with torch.no_grad():
        for step in range(10):

            dag, _, _ = controller()
            prec, loss = enas_validation_full_model(valid_queue, model, dag, nn.CrossEntropyLoss(), args, verbose=True)
            total_loss.update(loss, 1)
            total_top1.update(prec, 1)
            # if step % args.report_freq == 0:
            logging.info('eval %03d %e %f', step, loss, prec)
            logging.info('normal cell %s', str(dag[0].tolist()))
            logging.info('reduce cell %s', str(dag[1].tolist()))
            archs_pool.append(dag)
            archs_perf.append(prec)

    if return_perf:
        sorted_indices = np.argsort(archs_perf)[::-1]
        archs_pool = [archs_pool[i] for i in sorted_indices]
        archs_perf = [archs_perf[i] for i in sorted_indices]
        return total_top1.avg, archs_pool, archs_perf
    else:
        return total_top1.avg


def _query_model_by_eval(spec, nb_batch, model, valid_queue, test_queue, search_space:CNNSearchSpace,
                         criterion=None, args=None):
    """ Query one model by evaluate over some batchs"""
    valid_top1 = AverageMeter()
    test_top1 = AverageMeter()

    # change the model spec accordingly.
    model = search_space.change_model_spec(model, spec)

    with torch.no_grad():
        for q, top1 in zip([valid_queue, test_queue], [valid_top1, test_top1]):
            for ind, (input_tensor, target_tensor) in enumerate(q):
                if ind > nb_batch:
                    break
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
                logits, _ = model(input_tensor)
                # loss = criterion(logits, target_tensor)
                prec1, prec5 = accuracy(logits, target_tensor, topk=(1, 5))
                n = input_tensor.size(0)
                # objs.update(loss.item(), n)
                top1.update(prec1.item(), n)

    return {'validation_accuracy': valid_top1.avg, 'test_accuracy': test_top1.avg}


def _next_batches_repeat_loader(dataloader:RepeatedDataLoader, num_batches):
    queue = []
    for _ in range(num_batches):
        queue.append(dataloader.next_batch())
    return queue


def next_batches(dataloader, num_batches):
    queue = []
    _batch_count = 0
    for data in dataloader:
        _batch_count += 1
        queue.append(data)
        if _batch_count > num_batches:
            # process iteration
            break
    return queue


def compute_state_dict_norm(state_dict):
    sum_norm = 0
    with torch.no_grad():
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and 'conv' in k and 'weight' in k:
                sum_norm += torch.norm(v).item()
    return sum_norm


def _query_model_with_train_further(spec, nb_batch, model, train_queue, valid_queue, test_queue, policy):
    """
    Not used outside.
    So, we could clearly see a fact:

        model_state_dict is not resumable because of this.

    :param spec:
    :param nb_batch:
    :param model:
    :param train_queue:
    :param valid_queue:
    :param test_queue:
    :param policy:
    :return:
    """
    # Training further steps.
    backup_weights = clone_detach_model_state_dict(model)
    # norm_w = compute_state_dict_norm(backup_weights)
    nb = policy.args.neweval_num_train_batches
    nb_epoch = policy.args.neweval_num_train_epoch
    logging.info('Train {} batches for model_id {} before eval'.format(nb, spec))
    tr_epoch = []
    tr_acc = []
    tr_obj = []

    # original valid-res
    res_before = _query_model_by_eval(spec, nb_batch, model, valid_queue, test_queue, search_space=policy.search_space)

    for epoch in range(nb_epoch):
        # IPython.embed()
        _train_queue = next_batches(train_queue, nb)
        _batch_count = len(_train_queue)
        lr = policy.scheduler.get_lr()[0]
        if policy.args.debug:
            org_train_acc, org_train_obj = policy.eval_fn(_train_queue, policy.parallel_model,
                                                          criterion=policy._loss, verbose=policy.args.debug)
        else:
            org_train_acc, org_train_obj = 0., 0.

        train_acc, train_obj = darts_train_model(
            _train_queue, None, policy.parallel_model, policy.criterion, policy.optimizer, lr, policy.args,
            architect=None)

        # clean up the train queue completely.
        for d in _train_queue:
            del d
        _train_queue = []  # clean the data, destroy the graph.

        logging.info('-> epoch {}: Train acc {} -> {} | train obj {} -> {} '.format(
            epoch, org_train_acc, train_acc, org_train_obj, train_obj))
        tr_epoch.append(epoch)
        tr_acc.append(train_acc)
        tr_obj.append(train_obj)

    res = _query_model_by_eval(spec, nb_batch, model, valid_queue, test_queue, search_space=policy.search_space)
    res.update(
        train_steps=tr_epoch,
        train_accs=tr_acc,
        train_objs=tr_obj,
        validation_accuracy_before=res_before['validation_accuracy'],
        test_accuracy_before=res_before['test_accuracy'],
    )

    model.load_state_dict(backup_weights)
    return res


