"""
This is a test case to isolate the search space.

# Reference: Single Path One-Shot Neural Architecture Search with Uniform Sampling
# NOTE: Release this code after, build as fast as possible.
# Goal is to develop this and test on NasBench-101, small search space.
# Then to other search space. but now all the experiment is  NASBench-101 centric.
"""

import logging
import IPython

import numpy as np
import torch
from functools import partial
from collections import namedtuple, OrderedDict, deque

import utils
from nasws.cnn.policy.cnn_general_search_policies import CNNSearchPolicy
from nasws.cnn.policy.enas_policy import RepeatedDataLoader
from nasws.cnn.search_space.nasbench101.nasbench_search_space import NASbenchSearchSpace, \
    NasBenchSearchSpaceLinear, NasBenchSearchSpaceSubsample, NasBenchSearchSpaceICLRInfluenceWS, NasBenchSearchSpaceFixChannels

from nasws.cnn.search_space.nasbench101.model_search import NasBenchNetSearch
import nasws.cnn.procedures as procedure_ops
from nasws.cnn.search_space.nasbench101.util import change_model_spec
from nasws.cnn.utils import AverageMeter

Rank = namedtuple('Rank', 'valid_acc valid_obj geno_id gt_rank')


class NasBenchWeightSharingPolicy(CNNSearchPolicy):
    r""" Class to train NAS-Bench with weight sharing in general.

    This support all variation of NASBench space.
    Use this to build a simple network trainer and ranking, to facilitate other usage.

    Super class this to support many other things.
    """

    # defining the model spec placeholder.
    model_spec = None
    model_spec_id = None

    ## This only belongs to NASbench legacy class
    @property
    def nasbench_model_specs(self):
        return self.search_space.nasbench_hashs

    @property
    def nasbench_hashs(self):
        return self.search_space.nasbench_hashs

    def model_spec_by_id(self, mid):
        return self.search_space.nasbench_model_specs[mid]

    def __init__(self, args, full_dataset=False):
        """ Init and ad this. """
        self.args = args
        self.full_dataset = full_dataset
        super(NasBenchWeightSharingPolicy, self).__init__(
            args=args,
            sub_dir_path='{}_SEED_{}'.format(args.supernet_train_method, args.seed)
        )
        self.counter = 0
        self._change_model_fn = change_model_spec

    def initialize_search_space(self):
        args = self.args

        if args.search_space == 'nasbench101':
            self.model_fn = NasBenchNetSearch
            self.search_space = NASbenchSearchSpace(args, full_dataset=self.full_dataset)
        elif args.search_space == 'nasbench101_linear':
            self.model_fn = NasBenchNetSearch
            self.search_space = NasBenchSearchSpaceLinear(args)
        elif args.search_space == 'nasbench101_subsample':
            self.model_fn = NasBenchNetSearch
            self.search_space = NasBenchSearchSpaceSubsample(args)
        elif args.search_space == 'nasbench101_iclr_wsinfluence':
            self.model_fn = NasBenchNetSearch
            self.search_space = NasBenchSearchSpaceICLRInfluenceWS(args)
        elif args.search_space == 'nasbench101_fix_channels':
            self.model_fn = NasBenchNetSearch
            self.search_space = NasBenchSearchSpaceFixChannels(args)
        else:
            raise NotImplementedError("Other search space not supported at this moment.")


class NasBenchWeightSharingNewEval(NasBenchWeightSharingPolicy):
    """
    Testing the idea of, before evaluation, train a few batches
    """
    eval_result = {}
    ranking_per_epoch_before = OrderedDict()

    def __init__(self, args):
        super(NasBenchWeightSharingNewEval, self).__init__(args)
        self.eval_train_fn = partial(procedure_ops.darts_train_model, args=self.args, architect=None, sampler=None)

    def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None):
        # This just directly use this extra steps. May need new debugging.
        return procedure_ops.evaluate_extra_steps(self, epoch, data_source, fitnesses_dict, train_queue)

    def save_results(self, epoch, rank_details=True):
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
            'rank_per_epoch_before': self.ranking_per_epoch_before,
            'trained_model_spec_per_steps': self.trained_model_spec_ids,
            'eval_result': self.eval_result
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details, compute_kdt_before=True,
                                  sparse_kdt=True, percentile=True, random_kdt=True)


class NasBenchNetOneShotPolicy(NasBenchWeightSharingPolicy):
    """
    This is implemented with NAO implementation, i.e. add this arch pool idea.

    Only support nasbench search space for now. Add more later.

    Use this to build a simple network trainer and ranking, to facilitate other usage.

    """
    trained_model_spec_ids = []
    eval_result = OrderedDict()

    def __init__(self, args):
        super(NasBenchNetOneShotPolicy, self).__init__(args=args)
        # initialize eval pool. TODO Check if this cause the trouble, yes it is causing trouble now.
        self.search_space.evaluate_model_spec_ids = deque()
        arch_ids = self.search_space.random_eval_ids((args.num_intermediate_nodes - 1) * 40)
        for arch_id in arch_ids:
            self.search_space.eval_model_spec_id_append(arch_id)
        logging.info("Initial pool size {}".format(self.search_space.evaluate_model_spec_id_pool()))

    def run(self):
        """
        Difference with super.run() is, it will change the eval pool by random sampling some new architecture to replace
        the old ones.
        :return:
        """
        train_queue, valid_queue, test_queue, criterion = self.initialize_run()
        repeat_valid_queue = RepeatedDataLoader(valid_queue)
        args = self.args
        model, optimizer, scheduler = self.initialize_model()
        fitness_dict = {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        logging.info(">> Begin the search with supernet method: {}".format(args.supernet_train_method))
        logging.info("Always setting BN-train to True!")
        for epoch in range(self.epoch, args.epochs):
            args.current_epoch = epoch
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            # training for each epoch.
            train_acc, train_obj = self.train_fn(train_queue, valid_queue, model, criterion, optimizer, lr)
            self.logging_fn(train_acc, train_obj, epoch, 'Train', display_dict={'lr': lr})

            # do this after pytorch 1.1.0
            scheduler.step()

            # validation, compare to the traditional, we only evaluate the eval arch pool in this step.
            validate_accuracies, valid_acc, valid_obj = self.child_valid(
                model, repeat_valid_queue, self.search_space.evaluate_model_spec_id_pool(), criterion)

            self.logging_fn(valid_acc, valid_obj, epoch, 'Valid')
            utils.save_checkpoint(model, optimizer, self.running_stats, self.exp_dir, scheduler)

            if not self.check_should_save(epoch):
                continue
            self.save_duplicate_arch_pool('valid', epoch)
            logging.info("Evaluating and save the results.")
            logging.info("Totally %d architectures now to evaluate",
                         len(self.search_space.evaluate_model_spec_id_pool()))

            # evaluate steps.
            fitness_dict = self.evaluate(epoch, test_queue, fitnesses_dict=fitness_dict, train_queue=train_queue)
            # Generate new archs for robust evaluation.
            # replace bottom archs
            num_new_archs = self.search_space.replace_eval_ids_by_random(args.controller_random_arch)
            logging.info("Generate %d new archs", num_new_archs)
            self.save_results(epoch, rank_details=True)

        # add later, return the model specs that is evaluated across the time.
        # Process the ranking in the end, return the best of training.
        # ep_k = [k for k in self.ranking_per_epoch.keys()][-1]
        # best_id = self.ranking_per_epoch[ep_k][-1][1].geno_id
        # return best_id, self.nasbench_model_specs[best_id]
        return self.post_search_phase()

    def save_results(self, epoch, rank_details=False):
        # IPython.embed(header='Checking save results json problem')
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
            'trained_model_spec_per_steps': self.trained_model_spec_ids,
            'top-k-left': list(self.search_space.evaluate_model_spec_id_pool())
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details, sparse_kdt=True, percentile=True, random_kdt=True)

    def child_valid(self, model, valid_queue, arch_pool, criterion):
        valid_acc_list = []
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        logging.info("num valid arch {}".format(len(arch_pool)))
        with torch.no_grad():
            model.eval()
            for i, arch in enumerate(arch_pool):
                # for step, (inputs, targets) in enumerate(valid_queue):
                inputs, targets = valid_queue.next_batch()
                inputs = inputs.cuda()
                targets = targets.cuda()
                n = inputs.size(0)
                arch_l = arch
                
                model = change_model_spec(model, self.search_space.topologies[arch])
                # model.module.unused_modules_off()
                logits, _ = model(inputs)
                loss = criterion(logits, targets)
                prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
                valid_acc_list.append(prec1.data / 100)
                # model.module.unused_modules_back()
                if (i + 1) % 100 == 0:
                    logging.info('Valid arch %s loss %.2f top1 %f top5 %f',
                                 self.search_space.process_archname_by_id(arch_l),
                                 loss, prec1, prec5)

        self.save_arch_pool_performance(arch_pool, valid_acc_list, prefix='valid')
        return valid_acc_list, objs.avg, top1.avg

    @staticmethod
    def _compute_kendall_tau(ranking_per_epoch, compute_across_time=False):
        return NasBenchWeightSharingPolicy._compute_kendall_tau(ranking_per_epoch, compute_across_time=False)
