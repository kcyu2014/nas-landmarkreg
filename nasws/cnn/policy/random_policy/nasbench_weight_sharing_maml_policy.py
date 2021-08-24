"""
Implement as MAML policy.

    Extend the previous implemented new-eval as MAML.

"""
from collections import OrderedDict

import logging

import IPython
import torch
from functools import partial

import utils
from nasws.cnn.procedures.utils_maml import initialize_model_lr_by_named_module_parameters, \
    module_parameters_to_param_gen, eval_model_named_params
from nasws.cnn.policy.random_policy import NasBenchNetOneShotPolicy
import nasws.cnn.procedures as procedure_ops

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


class MAMLPlusNasBenchPolicy(NasBenchNetOneShotPolicy):
    """
    Implementation of the improved MAML for nasbench training. Ideally, it should be clearly separated.
    """
    def __init__(self, args):
        print("MAML++ policy For NasBench")
        super(MAMLPlusNasBenchPolicy, self).__init__(args)

        # use MAML_Plus train function
        self.args.supernet_train_method = 'maml-plus'
        self.train_fn = partial(procedure_ops.mamlplus_nas_weight_sharing_epoch, args=args,
                                policy=self, sampler=self.random_sampler, architect=None)
        self.named_learning_rate = None   # to be initialized based on the meta network.
        self.ranking_per_epoch_before = OrderedDict()

    def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None):
        # This just directly use this extra steps. May need new debugging.
        return procedure_ops.mamlplus_evaluate_extra_steps(self, epoch, data_source, fitnesses_dict, train_queue)

    def save_results(self, epoch, rank_details=True):
        dynamic_learning_rate = {k: v for k, v in module_parameters_to_param_gen(
            eval_model_named_params(
                self.named_learning_rate,
                fn=lambda x: x.cpu().numpy().tolist()
            ), with_name=True
        )}

        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
            'rank_per_epoch_before': self.ranking_per_epoch_before,
            'trained_model_spec_per_steps': self.trained_model_spec_ids,
            'eval_result': self.eval_result,
            'dynamic_lr': dynamic_learning_rate
        }

        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details, compute_kdt_before=True)

    def trainable_parameters(self):
        # return itertools.chain(self.model.parameters(), self.names_learning_rate_dicts.values())
        for it in [self.model.parameters(), module_parameters_to_param_gen(self.named_learning_rate)]:
            for p in it:

                if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                    yield p
                if isinstance(p, torch.nn.ParameterDict):
                    for pp in p.values():
                        yield pp

    def initialize_model(self):
        """
        Add initi
        :return:
        """
        args = self.args
        model = self.model_fn(args)
        # parameter dict?? may be replaced.
        self.named_learning_rate = initialize_model_lr_by_named_module_parameters(model, args)

        # multi parameters
        if args.gpus > 0:
            if self.args.gpus == 1:
                model = model.cuda()
                self.parallel_model = model
            else:
                self.model = model
                self.parallel_model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.parallel_model = model
        # rewrite the pointer
        model = self.parallel_model

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        logging.info("Using Adam for MAML ++")
        logging.info('Setting detect_anomaly == True')
        torch.autograd.set_detect_anomaly(True)

        # SGD optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)
        optimizer.add_param_group(
            {
                'params': list(module_parameters_to_param_gen(self.named_learning_rate, expose_lr=True)),
                'lr': args.learning_rate_min,
            }
        )
        logging.info('Optimizer: model parameters learning rate {} and DynamicLR learning rate {}'.format(
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))

        # scheduler as Cosine.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        return model, optimizer, scheduler

