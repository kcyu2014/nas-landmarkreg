# define this MAML optimizer, instead of the traditional optimizer.
# make this agnostic to original code.
import gc
import operator

import torch
import torch.nn as nn

import logging

from nasws.cnn.procedures.utils_maml import get_per_step_loss_importance_vector, \
    check_is_leaf_parameters, delete_module_params, eval_model_named_params
from nasws.cnn.search_space.nasbench101.util import change_model_spec, nasbench_model_forward
from visualization.process_data import tensorboard_summarize_list
from .utils_maml import named_module_parameters, clone_parameters, assign_parameters, NonLeafSGD
from nasws.cnn.policy.enas_policy import RepeatedDataLoader
from nasws.cnn.utils import AverageMeter, accuracy, Rank


def _summarize_shared_train(curr_step, meta_loss,
                            task_loss,
                            acc=0, acc_5=0, lr=0.0,
                            task_acc=0, task_acc5=0,
                            epoch_steps=1,
                            writer=None):
    """Logs a set of training steps."""
    # cur_loss = utils.to_item(total_loss) / epoch_steps
    # task_loss = utils.to_item(total_loss) / epoch_steps
    # cur_raw_loss = utils.to_item(raw_total_loss) / epoch_steps

    logging.info(f'| episode {curr_step:3d} '
                 f'| lr {lr:4.2f} '
                 f'| meta loss {meta_loss:.2f} '
                 f'| acc {acc:8.2f} '
                 f'| acc-5 {acc_5: 8.2f} '
                 f'|| task loss {task_loss:.2f} '
                 f'| task acc {task_acc: 8.4f} '
                 f'| task acc-5 {task_acc5: 8.4f}'
                 )

    # Tensorboard
    if writer is not None:
        writer.scalar_summary('shared/meta_loss',
                              meta_loss,
                              epoch_steps)
        writer.scalar_summary('shared/meta_accuracy',
                              acc,
                              epoch_steps)
        writer.scalar_summary('shared/task_loss',
                              task_loss,
                              epoch_steps)
        writer.scalar_summary('shared/task_accuracy',
                              task_acc,
                              epoch_steps)


def task_update_step(model, data, task_lr, criterion, args,
                     objs=None, top1=None, top5=None):
    """
    Train the model on a single few-shot task.
    We train the model with single or multiple gradient update.

    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        task_lr: (float) a task-specific learning rate
        criterion: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of
                     support set and query set
        args: (Params) hyperparameters
        create_graph: if we want approx-MAML, change create_graph=True to False
    """

    input, target = data
    # meta_params = named_module_parameters(model)
    clone_params = clone_parameters(model, return_pointer=True)
    assign_parameters(model, clone_params)

    # this is the task parameters, which is a clone of original one, but still inside the computational graph.
    optimizer = NonLeafSGD(model.parameters(), lr=task_lr)
    n = input.size(0)

    for _ in range(0, args.maml_num_train_updates):
        optimizer.zero_grad()
        logits, aux_logits = model(input)
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += aux_loss * 0.4

        optimizer.compute_loss_nonleaf_step(loss, args.grad_clip, args.maml_second_order)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        if objs:
            objs.update(loss.item(), n)
        if top1:
            top1.update(prec1.item(), n)
        if top5:
            top5.update(prec5.item(), n)

    # assign_parameters(model, meta_params)
    return clone_params


def maml_nas_weight_sharing(meta_queue, task_queue, model, criterion, optimizer, lr, args,
                            sampler=None, architect=None, valid_queue=None,
                            ):
    """
    This defines a traditional "Epoch", by runs out of training data.

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
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    task_objs = AverageMeter()
    task_top1 = AverageMeter()
    task_top5 = AverageMeter()

    # parameters
    # num_classes = args.maml_num_classes
    # num_samples = args.maml_num_samples
    # num_query = args.maml_num_query
    # meta_lr = args.maml_meta_lr should be outside metaOptimizer but not inside
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
            _summarize_shared_train(episode, objs.avg, task_objs.avg, top1.avg, top5.avg,
                                    task_lr, task_top1.avg, task_top5.avg)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        # conclude one MAML update loop.
        # IPython.embed(header='Checking memory is released for next episode loop')
        # Checked with the memory, after entire loop, it is correctly cleaned the pointers.

    return (top1.avg, task_top1.avg), (objs.avg, task_objs.avg)


##########################################
#           MAML++                       #
##########################################


def mamlplus_task_update_step(model, data, task_lr, criterion, args,
                              step_idx, clone_params, epoch_idx=0,
                              objs=None, top1=None, top5=None):
    """
    Perform MAML++ task-update step.

    TODO support BatchNorm and LayerNorm backup status.
    Note, it will not change the model's parameter. returns a new
    Args:
        model: (MetaLearner) a meta-learner to be adapted for a new task
        task_lr: (float) a task-specific learning rate
        criterion: a loss function
        dataloaders: (dict) a dict of DataLoader objects that fetches both of
                     support set and query set
        args: (Params) hyperparameters
        create_graph: if we want approx-MAML, change create_graph=True to False
    :return: Dict() as module_parameters, containing the updated version of parameters, in torch.Tensor()
    """

    input, target = data
    n = input.size(0)
    loss, logits, aux_logits = nasbench_model_forward(model, input, target, criterion)
    named_parameters = dict(model.module.named_parameters())
    use_second_order = args.maml_second_order and epoch_idx > args.mamlplus_first_order_to_second_order_epoch
    grads = torch.autograd.grad(loss, named_parameters.values(),
                                create_graph=use_second_order, allow_unused=True)
    named_grads = dict(zip(named_parameters.keys(), grads))
    update_clone_params = dict()
    if not args.mamlplus_dynamic_lr_relu:
        lr_fn = lambda x: x
    else:
        lr_fn = lambda x: torch.nn.functional.relu(x) + args.mamlplus_dynamic_lr_min

    for m_name, module_params in clone_params.items():
        update_clone_params[m_name] = dict()
        for p_name in module_params.keys():
            grad = named_grads[m_name + '.' + p_name]
            if grad is not None:
                g_d = -grad.data * lr_fn(task_lr[m_name][p_name][str(step_idx)])
                update_clone_params[m_name][p_name] = clone_params[m_name][p_name] + g_d
    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    if objs:
        objs.update(loss.item(), n)
    if top1:
        top1.update(prec1.item(), n)
    if top5:
        top5.update(prec5.item(), n)
    return update_clone_params


def mamlplus_nas_weight_sharing_epoch(meta_queue, task_queue, model, criterion, optimizer, lr, args, policy,
                                      sampler=None, architect=None, valid_queue=None,
                                      ):
    """
    This defines a traditional "Epoch", by runs out of training data.
    TODO: Better clone parameters.
        Because that, we only clone parameter for each "sampled architecture", so, it is idea that we just
        use parameter cloning only for this arch.
        i.e. Per-architecture clones, and updates, i.e.i.e. allow_unused=False.

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
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    task_objs = AverageMeter()
    task_top1 = AverageMeter()
    task_top5 = AverageMeter()

    if not isinstance(task_queue, RepeatedDataLoader):
        task_queue = RepeatedDataLoader(task_queue)

    num_inner_tasks = args.maml_num_inner_tasks

    num_episodes = len(meta_queue) // num_inner_tasks
    check_is_leaf_parameters(model)
    meta_parameters = named_module_parameters(model)

    task_lr = policy.named_learning_rate
    logging.info(f"Epoch episode = {num_episodes}")
    meta_queue = iter(meta_queue)
    epoch_idx = policy.epoch

    for episode in range(num_episodes):
        if args.debug and episode > 2:
            logging.warning('Testing only, break after 2 episodes.')
            break

        meta_loss = 0.0
        total_model = 0
        optimizer.zero_grad()
        # IPython.embed(header='Checking memory using before episode start')
        # TODO 1264Mb before this start, node = 3
        # Aggregate over num_inner_tasks sub-graph.

        for n_task in range(num_inner_tasks):
            # For each task, clone a set of parameter.
            input, target = task_queue.next_batch()
            input = input.cuda().requires_grad_()
            target = target.cuda()
            meta_input, meta_target = next(meta_queue)
            n = meta_input.size(0)
            meta_input = meta_input.cuda().requires_grad_()
            meta_target = meta_target.cuda()

            if sampler:     # sample an architecture for each task
                model = sampler(model, architect, args)
                total_model += 1
            per_step_loss_importance_vectors = get_per_step_loss_importance_vector(epoch_idx, args)
            clone_params = clone_parameters(model, return_pointer=True)

            t_loss = 0
            for step_idx in range(args.mamlplus_number_of_training_steps_per_iter):
                # Task-update the parameters, clone the parameters.
                clone_params = mamlplus_task_update_step(
                    model, (input, target), task_lr, criterion, args,
                    step_idx, clone_params, epoch_idx, task_objs, task_top1, task_top5)
                # make sure the clone params is assigned for meta-loss computation.
                assign_parameters(model, clone_params) # move the assigned parameter back.

                # compute the meta_loss and do grad backward
                if args.mamlplus_use_multi_step_loss_optimization and epoch_idx < args.mamlplus_multi_step_loss_num_epochs:
                    loss, meta_logits, meta_aux_logits = nasbench_model_forward(
                        model, meta_input, meta_target, criterion)
                    loss = per_step_loss_importance_vectors[step_idx] * criterion(meta_logits, meta_target)
                else:
                    if step_idx == args.mamlplus_number_of_training_steps_per_iter - 1:
                        loss, meta_logits, meta_aux_logits = nasbench_model_forward(
                            model, meta_input, meta_target, criterion)
                    else:
                        continue

                # loss.backward(retain_graph=True)
                t_loss = loss + t_loss
                # keep computing the backward gradients throught the time.
                meta_loss = meta_loss + loss.item()
                objs.update(loss.item(), n)
                prec1, prec5 = accuracy(meta_logits.detach(), meta_target, topk=(1,5))
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
            t_loss.backward()
            assign_parameters(model, meta_parameters)
            delete_module_params(clone_params)
            gc.collect()

            # IPython.embed(header='check end of task loop.')
            # 1st: 2345 (almost doubled), 790 MB
            # 2nd: 2345, 790 MB, no leak!
            # 3rd: 2347,
            # 5th: 2351MB.  -> 3120MB -> 3123
            # end step, delete intermediate clone-parameters and assign the original parameter back.

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        # conclude one MAML update loop.
        if episode % args.report_freq == 0:
            _summarize_shared_train(episode, objs.avg, task_objs.avg, top1.avg, top5.avg,
                                    lr, task_top1.avg, task_top5.avg)
        # IPython.embed(header='Checking memory is released for next episode loop')
        # Checked with the memory, after entire loop, it is correctly cleaned the pointers.
        # 1st: 3094MB -> 3115MB

    return (top1.avg, task_top1.avg), (objs.avg, task_objs.avg)


def mamlplus_eval_train_model(train_queue, model,  criterion, task_lr, args, epoch_idx,):
    """
    MAML++ Evaluate training steps. This is the same as before, only difference is No meta update.
    NOTE: This module is checked and will not increase the memory after calling it.
    :param train_queue:
    :param model:
    :param criterion:
    :param task_lr:
    :param args:
    :param epoch_idx:
    :return: (obj, acc), new_param state dict.
    """
    task_objs = AverageMeter()
    task_top1 = AverageMeter()
    task_top5 = AverageMeter()
    # IMPORTANT! fixing memory leak. should not copy, but directly update the parameters.
    # We can do it because we backup the original weights and will be restored later.
    clone_params = named_module_parameters(model)

    for step, (input, target) in enumerate(train_queue):
        model.train()
        if args.debug and step > 10:
            logging.warning('Testing only. Break after 10 batches.')
            break

        input = input.cuda().requires_grad_()
        target = target.cuda()
        for step_idx in range(args.mamlplus_number_of_training_steps_per_iter):
            # Task-update the parameters, clone the parameters.
            clone_params = mamlplus_task_update_step(
                model, (input, target), task_lr, criterion, args,
                step_idx, clone_params, epoch_idx, task_objs, task_top1, task_top5)
            assign_parameters(model, clone_params)  # move the assigned parameter back.

    new_weights_dict = model.state_dict()
    return (task_objs.avg, task_top1.avg), new_weights_dict


def mamlplus_evaluate_extra_steps(policy, epoch, data_source, fitnesses_dict=None, train_queue=None):
    """
    Full evaluation of all possible models.
    # TODO this is highly duplicated with the original evaluate extra steps. consider update later.
    :param epoch:
    :param data_source:
    :param fitnesses_dict: Store the model_spec_id -> accuracy
    :return:
    """
    nb = policy.args.mamlplus_number_of_training_steps_per_iter
    assert nb > 0
    if not train_queue:
        raise ValueError("New evaluation scheme requires a training queue.")

    fitnesses_dict = fitnesses_dict or {}
    total_avg_acc = 0
    total_avg_obj = 0

    model = policy.parallel_model
    # rank dict for the possible solutions
    model_specs_rank = {}
    model_specs_rank_before = {}

    # make sure the backup weights on CPU, do not occupy the space
    backup_weights = model.cpu().state_dict()
    model.cuda()

    _train_iter = enumerate(train_queue)  # manual iterate the data here.
    _train_queue = []
    # as backup
    ind = 0
    eval_before_train = {}
    eval_after_train = {}
    eval_pool = policy.evaluate_model_spec_id_pool()
    meta_params = named_module_parameters(model)

    # wrap the dynamic learning rate into an evaluation mode. Will not compute grad graph.
    eval_named_lr = eval_model_named_params(policy.named_learning_rate)
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
            change_model_spec(policy.parallel_model, new_model_spec)

            # Reset the weights.
            logging.debug('Resetting parallel model weights ...')
            # IPython.embed(header='start evaluating one model')

            model.load_state_dict(backup_weights)
            model.cuda()  # make sure this is on GPU.
            _avg_val_acc_before, _avg_val_obj_before = policy.eval_fn(data_source, policy.parallel_model,
                                                                      criterion=policy._loss, verbose=False)
            eval_before_train[model_spec_id] = _avg_val_acc_before, _avg_val_obj_before

            # re-train for a few batches.
            _train_queue = policy.next_batches(train_queue, nb)
            _batch_count = len(_train_queue)
            logging.debug('Train {} batches for model_id {} before eval'.format(_batch_count, model_spec_id))
            org_train_acc, org_train_obj = policy.eval_fn(_train_queue, policy.parallel_model,
                                                          criterion=policy._loss, verbose=policy.args.debug)

            # replace this part with
            (train_obj, train_acc), new_weights_dict = mamlplus_eval_train_model(
                _train_queue, policy.parallel_model,
                policy._loss, eval_named_lr, policy.args, policy.epoch)
            assign_parameters(model, meta_params)
            logging.debug('loading new weights dict')
            model.load_state_dict(new_weights_dict)

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
            # IPython.embed(header=f'Model {model_spec_id}: finish and move to next one, check GPU release')
            # update the total loss.
            total_avg_acc += _avg_val_acc
            total_avg_obj += _avg_val_obj

            # saving the particle fit in our dictionaries
            fitnesses_dict[model_spec_id] = _avg_val_acc
            ms_hash = policy.search_space.hashs[model_spec_id]
            model_specs_rank[ms_hash] = Rank(_avg_val_acc, _avg_val_obj, model_spec_id,
                                             policy.search_space.rank_by_mid[model_spec_id])
            model_specs_rank_before[ms_hash] = Rank(_avg_val_acc_before, _avg_val_obj_before, model_spec_id,
                                                    policy.search_space.rank_by_mid[model_spec_id])

            # manual collect the non-used graphs.
            del new_weights_dict
            gc.collect()
            # IPython.embed(header=f'Model {model_spec_id}: finish training == checked 2031MB')
        except StopIteration as e:
            _train_iter = enumerate(train_queue)
            logging.debug("Run out of train queue, {}, restart ind {}".format(e, ind - 1))
            ind = ind - 1

    # IPython.embed(header="Checking the results.")
    # save the ranking, according to their GENOTYPE but not particle id
    rank_gens = sorted(model_specs_rank.items(), key=operator.itemgetter(1))
    rank_gens_before = sorted(model_specs_rank_before.items(), key=operator.itemgetter(1))
    # hash to positions mapping, before training
    rank_gens_before_pos = {elem[0]: pos for pos, elem in enumerate(rank_gens_before)}

    policy.ranking_per_epoch[epoch] = rank_gens
    policy.ranking_per_epoch_before[epoch] = rank_gens_before
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

