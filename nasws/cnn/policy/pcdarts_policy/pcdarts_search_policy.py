"""PCDarts result

"""
import os
import logging
from nasws.cnn.search_space.nasbench201.nasbench201_search_space import NASBench201SearchSpace
import torch.nn as nn

from ..differentiable_policy import DifferentiableCNNPolicy
from .architect import PCDartsArchitect
from prettytable import PrettyTable

def update_pcdarts_args(args, pcdarts_args):
    # wrap supernet_warmup_epoch as warmup_epoch
    pcdarts_args.warmup_epochs = args.supernet_warmup_epoch
    return pcdarts_args


class PCDARTSSearchPolicy(DifferentiableCNNPolicy):
    top_K_complete_evaluate = 200
    top_K_num_sample = 1000

    def __init__(self, args, pcdarts_args) -> None:
        super(PCDARTSSearchPolicy, self).__init__(args)
        pcdarts_args = update_pcdarts_args(args, pcdarts_args)
        self.pcdarts_args = pcdarts_args
        self.args.policy_args = pcdarts_args
        self.policy_epochs = pcdarts_args.epochs

    # otherwise, define the run profile
    def initialize_model(self):
        tmp_resume, self.args.resume = self.args.resume, False
        model, optimizer, scheduler = super().initialize_model()
        model.ALWAYS_FULL_PARAMETERS = True
        self.args.resume = tmp_resume
        # load architect
        # anyway this is simply a pointer to DataParallel..., always using self.model to avoid the problem of module. smt.
        self.architect = PCDartsArchitect(self.model, self.pcdarts_args, None) # as a wrapper to train the arch parames
        if self.args.resume:
            self.resume_from_checkpoint()
        return model, optimizer, scheduler
