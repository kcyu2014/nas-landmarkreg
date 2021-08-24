import torch 
import torch.nn as nn
import logging
from functools import partial
from ..differentiable_policy import DifferentiableCNNPolicy
from .architect import GDASArchitect
from .configs import gdas_args as policy_args

def adjust_tau_value(epoch, args):
    tau = args.tau_max - (args.tau_max-args.tau_min) * epoch / (args.epochs-1) 
    args.tmp_tau = tau
    return tau
 

class GDASCNNSearchPolicy(DifferentiableCNNPolicy):

    def __init__(self, args):
        super().__init__(args)
        self.policy_args = policy_args
        self.args.policy_args = policy_args

    def initialize_model(self, resume=False):
        model, optimizer, scheduler = super().initialize_model(resume=False)
        self.architect = GDASArchitect(self.model, self.args)
        if resume or self.args.resume:
            self.resume_from_checkpoint()
        return model, optimizer, scheduler

    def initialize_differentiable_policy(self, criterion):
        super().initialize_differentiable_policy(criterion)
        self.architect.module_forward_fn = partial(self.search_space.module_forward_fn, criterion=criterion)

    # flush
    @property
    def epoch(self):
        if 'epoch' in self.running_stats.keys():
            e = self.running_stats['epoch']
        else:
            # set this
            self.epoch = 0
            e = 0

        self.args.current_epoch = e
        self.args.gdas_tau = adjust_tau_value(e, self.policy_args)
        # set the global variable gdas_tau
        return e

    @epoch.setter
    def epoch(self, epoch):
        self.running_stats['epoch'] = epoch
        self.args.current_epoch = epoch
        self.args.gdas_tau = adjust_tau_value(epoch, self.policy_args)


