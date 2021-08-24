# Base code is from https://github.com/cs230-stanford/cs230-code-examples
import gc
import json
import logging
import os
import random

import IPython
import shutil
# from collections import OrderedDict

import torch
import matplotlib.pyplot as plt
import numpy as np

from nasws.cnn.utils import clip_grad_norm


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console NOTE
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(logging.Formatter('%(message)s'))
        # logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


# def save_checkpoint(state, is_best, checkpoint):
#     """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
#     checkpoint + 'best.pth.tar'
#     Args:
#         state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
#         is_best: (bool) True if it is the best model seen till now
#         checkpoint: (string) folder where parameters are to be saved
#     """
#     filepath = os.path.join(checkpoint, 'last.pth.tar')
#     if not os.path.exists(checkpoint):
#         print("Checkpoint Directory does not exist! Making directory {}".
#               format(checkpoint))
#         os.mkdir(checkpoint)
#     else:
#         # print("Checkpoint Directory exists! ")
#         pass
#     torch.save(state, filepath)
#     if is_best:
#         shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


# def load_checkpoint(checkpoint, model, optimizer=None):
#     """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
#     optimizer assuming it is present in checkpoint.
#     Args:
#         checkpoint: (string) filename which needs to be loaded
#         model: (torch.nn.Module) model for which the parameters are loaded
#         optimizer: (torch.optim) optional: resume optimizer from checkpoint
#     """
#     if not os.path.exists(checkpoint):
#         raise ("File doesn't exist {}".format(checkpoint))
#     checkpoint = torch.load(checkpoint)
#     model.load_state_dict(checkpoint['state_dict'])
#
#     if optimizer:
#         optimizer.load_state_dict(checkpoint['optim_dict'])
#
#     return checkpoint


def plot_training_results(model_dir, plot_history):
    """
    Plot training results (procedure) during training.

    Args:
        plot_history: (dict) a dictionary containing historical values of what
                      we want to plot
    """
    tr_losses = plot_history['train_loss']
    te_losses = plot_history['test_loss']
    tr_accs = plot_history['train_acc']
    te_accs = plot_history['test_acc']

    plt.figure(0)
    plt.plot(list(range(len(tr_losses))), tr_losses, label='train_loss')
    plt.plot(list(range(len(te_losses))), te_losses, label='test_loss')
    plt.title('Loss trend')
    plt.xlabel('episode')
    plt.ylabel('ce loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'loss_trend'), dpi=200)

    plt.figure(1)
    plt.plot(list(range(len(tr_accs))), tr_accs, label='train_acc')
    plt.plot(list(range(len(te_accs))), te_accs, label='test_acc')
    plt.title('Accuracy trend')
    plt.xlabel('episode')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'accuracy_trend'), dpi=200)


# checking purpose, if the two dict are same not.
def compare_model_dicts(d1, d2):
    for k, v in d1.items():
        if k in d2.keys():
            print('diff {}: {}'.format(k, torch.sum(v - d2[k]).item()))


def unwrap_model(parallel_model):
    if isinstance(parallel_model, torch.nn.DataParallel):
        _model = parallel_model.module
    else:
        _model = parallel_model

    return _model


def named_module_parameters(model):
    """
    Get the parameters from named-module.
    It is equal to copy(), the purpose is to return the pointer
    to each individual nn.Parameter inside a Module.

    :param _model:
    :return:
    """
    _model = unwrap_model(model)

    module_parameters = dict()
    for name, module in _model.named_modules():
        new_dict = dict()
        for k, v in module._parameters.items():  # basically it is equal to copy
            if isinstance(v, torch.Tensor):
                new_dict[k] = v
        module_parameters[name] = new_dict
    return module_parameters


def initialize_model_lr_by_named_module_parameters(model, args):
    """
    Initialize the corresponding learnable LR for MAML++ procedure.

    :param model:
    :return: parameterized lr with the same hierarchy of named_module_parameters(model).
    """
    init_task_lr = args.maml_task_lr
    named_learning_rate = dict()

    for name, module in model.named_modules():
        new_dict = dict()
        # new_dict = torch.nn.ParameterDict()
        for k, v in module._parameters.items():  # basically it is equal to copy
            if isinstance(v, torch.nn.Parameter):
                new_dict[k] = torch.nn.ParameterDict()
                for step_id in range(args.mamlplus_number_of_training_steps_per_iter + 1):
                    new_dict[k][str(step_id)] = torch.nn.Parameter(torch.ones(1).to('cuda:0') * init_task_lr)
                # new_dict[k] = torch.nn.Parameter(
                #     torch.ones(args.mamlplus_number_of_training_steps_per_iter + 1).to('cuda:0') * init_task_lr,
                # )
        named_learning_rate[name] = new_dict
    return named_learning_rate


def eval_model_named_params(module_params, fn=lambda x: x):
    eval_params = dict()
    for m_name, m_params in module_params.items():
        eval_params[m_name] = dict()
        for p_name, p in m_params.items():
            if isinstance(p, torch.Tensor):
                eval_params[m_name][p_name] = p.detach().clone()
            elif isinstance(p, torch.nn.ParameterDict):
                lr_dict = dict()
                for step_k in p.keys():
                    lr_dict[step_k] = fn(p[step_k].detach().clone())
                eval_params[m_name][p_name] = lr_dict
    return eval_params


def module_parameters_to_param_gen(module_parameters, with_name=False, expose_lr=False):
    """
    Map from module_parameters into model.named_parameters() format.
    :param module_parameters:
    :param with_name:
    :return:
    """
    for k, v in module_parameters.items():
        for k2, p in v.items():
            if expose_lr:
                assert isinstance(p, torch.nn.ParameterDict)
                for k3, pp in p.items():
                    if with_name:
                        yield k + '.' + k2 + '.' + k3, pp
                    else:
                        yield pp
            else:
                if with_name:
                    yield k + '.' + k2, p
                else:
                    yield p


def apply_fn_parameters(model, fn):
    """
    Apply the parameters.
    :param _model:
    :param fn:
    :return:
    """
    _model = unwrap_model(model)
    for name, module in _model.named_modules():
        for k, p in module._parameters.items():
            if p is not None:
                new_p = fn(p)
                if isinstance(new_p, torch.Tensor):
                    module._parameters[k] = new_p


def clone_parameters(model, return_pointer=False):
    """
    Clone the model parameters and replace the original.
    this is usually done before calling
    named_module_parameters, otherwise, it will destroy the
    original computational graph.

    :param model:
    :param return_pointer: True, return named_module_parameters(model_new)
    :return:
    """
    _model = unwrap_model(model)
    apply_fn_parameters(_model, lambda x: x.clone())
    if return_pointer:
        return named_module_parameters(_model)


def parameters_to(model, device='cuda:0'):
    _model = unwrap_model(model)
    apply_fn_parameters(_model, lambda x: x.to(device))


def check_is_leaf_parameters(model):
    _model = unwrap_model(model)
    def _check_is_leaf(x):
        if not x.is_leaf:
            IPython.embed()
            raise ValueError("Non leaf encountered.")

    apply_fn_parameters(_model, _check_is_leaf)


def assign_parameters(model, module_parameters):
    """
    Just assign the parameter pointers.
    :param model:
    :param module_parameters:
    :return:
    """
    _model = unwrap_model(model)
    for m_name, module in _model.named_modules():
        for p_name, p in module_parameters[m_name].items():
            module._parameters[p_name] = p


# Algorithm: MAML in NAS-WS.
# Input: SearchSpace, (distribution over tasks)
# Input: alpha, beta, as steps
class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"
required = _RequiredParameter()


class NonLeafSGD(torch.optim.SGD):
    """
    Non-leaf SGD, apply the SGD update to non-leaf parameters (or any tensors, that can be compute parameters.)
    It naturally extend SGD, so all setting is the same except accepting non-leaf tensors.

    It does not use grad, but use loss to compute
        grads = torch.autograd.grads(loss, parameters, create_graph=True)
    directly.

    """

    def step(self, closure=None):
        raise NotImplementedError("You should not call step for a non-leaf optimizer, "
                                  "use torch.optim.SGD directly")

    def compute_loss_nonleaf_step(self, loss, grad_clip=None, use_second_order=True):
        """
        :param loss:
        :param kwargs:
        :return:
        """

        # clear previous gradients, compute gradients of all variables wrt loss
        def zero_grad(params):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

        # loss = None
        # if closure is not None:
        #     loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # zero grad
            zero_grad(group['params'])

            # compute grads
            grads = torch.autograd.grad(loss, group['params'], create_graph=use_second_order, allow_unused=True)
            if grad_clip:
                clip_grad_norm(grads, grad_clip, norm_type=2)

            for p, grad in zip(group['params'], grads):
                if grad is None:
                    continue
                # assign gradient
                d_p = grad.data
                # p.data = p.data - group['lr'] * d_p
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # This should be a non-leaf node due to clone, but why it is not???
                p.add_(-group['lr'], d_p)
                # p.data.add_(-group['lr'], d_p)
                # IPython.embed()
        return loss

    def add_param_group(self, param_group):
        r"""
        Override the optimizer.add_param_group, to remove the non-leaf node one.

        Add a param group to the :class:`Optimizer` s `param_groups`.

                This can be useful when fine tuning a pre-trained network as frozen layers can be made
                trainable and added to the :class:`Optimizer` as training progresses.

                Arguments:
                    param_group (dict): Specifies what Tensors should be optimized along with group
                    specific optimization options.
                """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)


class NonLeafSGDAdaptedLR(object):
    """ Deprecated. This is moved to """

    def __init__(self, named_params, lr):
        super(NonLeafSGDAdaptedLR, self).__init__()
        self.named_params = named_params

    def check_is_leaf(self):
        for m_name, m_params in self.named_params.items():
            for p_name, p in m_params.items():
                if p.is_leaf:
                    print("WRONG!!! {}.{} is a leaf node.".format(m_name, p_name))

    def zero_grad(self):
        # zero_grad(self.named_params)
        for m_name, m_params in self.named_params.items():
            zero_grad(m_params)

    def compute_loss_nonleaf_step(self, loss, named_lr_dict, current_step_idx, grad_clip=None, use_second_order=True):

        named_parameters = dict(module_parameters_to_param_gen(self.named_params, with_name=True))
        grads = torch.autograd.grad(loss, named_parameters.values(),
                                    create_graph=use_second_order, allow_unused=True)
        named_grads = dict(zip(named_parameters.keys(), grads))

        for m_name, module_params in self.named_params.items():
            for p_name, p in module_params.items():
                grad = named_grads[m_name + '.' + p_name]
                if grad is not None:
                    assert isinstance(named_lr_dict[m_name][p_name], torch.nn.Parameter)
                    g_d = - grad.data * named_lr_dict[m_name][p_name][current_step_idx]
                    # g_d.mul_(-0.1)
                    # g_d.mul_(-named_lr_dict[m_name][p_name][current_step_idx])
                    self.named_params[m_name][p_name] = p + g_d
        # note this will change the saved dictionary.
        return self.named_params


def zero_grad(params):
    for name, param in params.items():
        if param.requires_grad == True:
            if param.grad is not None:
                if torch.sum(param.grad) > 0:
                    # print(param.grad)
                    param.grad.zero_()
                    params[name].grad = None


def meta_update(self, loss):
    """
    Applies an outer loop update on the meta-parameters of the model.
    :param loss: The current crossentropy loss.
    """
    self.optimizer.zero_grad()
    loss.backward()
    if 'imagenet' in self.args.dataset_name:
        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
    self.optimizer.step()


def get_per_step_loss_importance_vector(current_epoch, args):
    """
    Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
    loss towards the optimization loss.
    :return: A list of importance weights, to be used to compute the weighted average of the loss, useful for
    the MSL (Multi Step Loss) mechanism.
    Should be used a Tensor later.
    """
    loss_weights = np.ones(shape=(args.mamlplus_number_of_training_steps_per_iter)) * (
            1.0 / args.mamlplus_number_of_training_steps_per_iter)
    decay_rate = 1.0 / args.mamlplus_number_of_training_steps_per_iter / args.mamlplus_multi_step_loss_num_epochs
    min_value_for_non_final_losses = 0.03 / args.mamlplus_number_of_training_steps_per_iter
    for i in range(len(loss_weights) - 1):
        curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
        loss_weights[i] = curr_value

    curr_value = np.minimum(
        loss_weights[-1] + (current_epoch * (args.mamlplus_number_of_training_steps_per_iter - 1) * decay_rate),
        1.0 - ((args.mamlplus_number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    return loss_weights


def get_inner_loop_parameter_dict(params, args):
    """
    Returns a dictionary with the parameters to use for inner loop updates.
    :param params: A dictionary of the network's parameters.
    :return: A dictionary of the parameters to use for the inner loop optimization process.
    """
    param_dict = dict()
    for name, param in params:
        if param.requires_grad:
            if args.mamlplus_enable_inner_loop_optimizable_bn_params:
                param_dict[name] = param
            else:
                IPython.embed(header='check batch norm not update logic')
                if "bn" not in name:
                    param_dict[name] = param
    return param_dict


def delete_module_params(module_params):
    """
    Delete the pointer completely to release the memory.
    :param module_params:
    :return:
    """
    for m_name in list(module_params.keys()):
        for p_name in list(module_params[m_name].keys()):
            del module_params[m_name][p_name]
        del module_params[m_name]
    gc.collect()

