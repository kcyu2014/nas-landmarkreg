import os
from collections import namedtuple

import torch
import logging

# from visualize import plot

Rank = namedtuple('Rank', 'valid_ppl geno_id')


class CNNContinualTrainEvaluationPhase:
    """
    Train a specific epochs after, then store the results as comparison.


    Support: resume training from the beginning.

    Search Space for now: nasbench

    """

    def __init__(self, args, test=False):
        self.args = args
        self.start_epoch = 1  # Keep track the starting epoch
        self.exp_dir = None

        if test:
            self.initialize_test()
        else:
            self.initialize_run()

    def initialize_test(self):
        pass

    def initialize_run(self):
        # reload the path.
        pass

    def run(self):
        pass

    def evaluate(self, data_source, batch_size=10):
        pass

    def train(self, epoch):
        pass

    def test(self, model_path, use_logger=True):
        pass

    def rollback_to_previous_best_model(self):
        self.model = torch.load(os.path.join(self.exp_dir, 'model.pt'))
        self.parallel_model = self.model.cuda()

        optimizer_state = torch.load(os.path.join(self.exp_dir, 'optimizer.pt'))
        if 't0' in optimizer_state['param_groups'][0]:
            self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.args.lr, t0=0, lambd=0.,
                                              weight_decay=self.args.wdecay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wdecay)
        self.optimizer.load_state_dict(optimizer_state)
        self.epoch = torch.load(os.path.join(self.exp_dir, 'misc.pt'))['epoch']
        logging.info('rolling back to the previous best model at epoch {}...'.format(self.epoch))
