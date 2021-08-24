import logging
from abc import ABC, abstractmethod
import random
from collections import OrderedDict

import IPython
from scipy.stats import kendalltau

import utils
from .search_space import SearchSpace
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
# after import torch, add this
from visualization.best_configs import BEST_RANK_BY_GENOTYPE
from visualization.plot_rank_change import process_rank_data_node_2

torch.backends.cudnn.benchmark=True
import torch.backends.cudnn as cudnn


class RNNSearchPolicy(ABC):
    """
    Search policy for RNN model.
    """
    def __init__(self, args):
        self.args = args
        # metrics to track
        self.ranking_per_epoch = OrderedDict()
        self.search_space = SearchSpace(self.args)
        self.writer = None # this is for TensorboardX writer.

    @abstractmethod
    def run(self):
        pass

    def initialize_run(self):
        """
        utility method for directories and plots
        :return:
        """
        if not self.args.continue_train:

            self.sub_directory_path = '{}_SEED_{}_geno_id_{}_{}'.format(self.args.save,
                                                                        self.args.seed,
                                                                        self.args.genotype_id,
                                                                        time.strftime("%Y%m%d-%H%M%S")
                                                                        )
            self.exp_dir = os.path.join(self.args.main_path, self.sub_directory_path)
            utils.create_dir(self.exp_dir)

        if self.args.visualize:
            self.viz_dir_path = utils.create_viz_dir(self.exp_dir)

        # Set logger.
        self.logger = utils.get_logger(
            "train_search",
            file_handler=utils.get_file_handler(os.path.join(self.exp_dir, 'log.txt')))

        # Set the random seed manually for reproducibility.
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.set_device(self.args.gpu)
                cudnn.benchmark = True
                cudnn.enabled = True
                torch.cuda.manual_seed_all(self.args.seed)

    def validate_model(self, current_model, data_source, current_geno_id, current_genotype, batch_size=10):
        """
        Evaluate throughout the validation dataset.
        :param current_model:
        :param data_source:
        :param current_geno_id:
        :param current_genotype:
        :param batch_size:
        :return:
        """
        current_model.eval()
        logging.debug('Evaluating Genotype {id} = {t}'.format(id=current_geno_id, t=current_genotype))
        total_loss = 0
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(batch_size)

        # chosing random batch to evaluate all the particles
        # self.logger.info('len valid data: {} bathces'.format(data_source.size(0) // self.args.bptt))
        # batch = np.random.randint(0, (data_source.size(0) // self.args.bptt))

        # Looping the entire dataset.
        for i in range(0, data_source.size(0) - 1, self.args.bptt):
            data, targets = utils.get_batch(data_source, i, self.args, evaluation=True)
            targets = targets.view(-1)
            log_prob, hidden = self.model(data, hidden)
            loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
            total_loss += loss * len(data)
            hidden = utils.repackage_hidden(hidden)

        avg_valid_loss = utils.to_item(total_loss) / len(data_source)
        avg_valid_ppl = math.exp(avg_valid_loss)

        return avg_valid_loss, avg_valid_ppl

    def _save_ranking_results(self, save_data, epoch):
        """
        Save the ranking results if necessary.
        :param save_data:
        :param epoch:
        :return:
        """
        # Plot the ranking data
        fig = process_rank_data_node_2(save_data,
                                       os.path.join(self.exp_dir, 'rank_change-{}.pdf'.format(epoch)))

        # Compute Kendall tau for every epochs and save them into result.
        epoch_keys = [k for k in reversed(self.ranking_per_epoch.keys())]
        epoch_keys.insert(0, 10000000)
        kd_tau = {}
        for ind, k in enumerate(epoch_keys[:-1]):
            elem = []
            if ind == 0 and self.args.num_intermediate_nodes == 2:
                rank_1 = np.array(BEST_RANK_BY_GENOTYPE, dtype=np.uint)
            else:
                rank_1 = np.array([elem[1].geno_id for elem in self.ranking_per_epoch[k]], dtype=np.uint)
            for j in epoch_keys[ind + 1:]:
                rank_2 = np.array([elem[1].geno_id for elem in self.ranking_per_epoch[j]], dtype=np.uint)
                elem.append(kendalltau(rank_1, rank_2))
            kd_tau[k] = elem
        logging.info("Latest Kendall Tau (ground-truth vs {}): {}".format(epoch_keys[1], kd_tau[10000000][0]))

        save_data['kendaltau'] = kd_tau
        # IPython.embed(header='Tensorboard')
        if self.writer is not None:
            self.writer.add_scalar('kendall_tau', kd_tau[10000000][0].correlation, epoch)
            self.writer.add_figure(tag='rank-diff', figure=fig, global_step=epoch)
        return save_data

    def _save_results(self, save_data, epoch, rank_details=False):

        if rank_details:
            save_data = self._save_ranking_results(save_data, epoch)

        utils.save_json(save_data, os.path.join(self.exp_dir, 'result.json'))
        # if hasattr(self, 'writer') and self.writer is not None:
        #     self.writer.export_scalars_to_json(os.path.join(self.exp_dir, 'tb_scalars.json'))

    def save_results(self, epoch, rank_details=False):
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details)


class RandomRNNSearchPolicy(RNNSearchPolicy):

    def __init__(self, args):
        super(RandomRNNSearchPolicy, self).__init__(args=args)

        self.range_possible_geno_ids = math.factorial(self.args.num_intermediate_nodes) * (
                    self.args.num_operations ** self.args.num_intermediate_nodes)

    def run(self):

        random.seed(self.args.seed)

        genotype_id = random.randint(0, self.range_possible_geno_ids)
        genotype = self.search_space.genotype_from_id(genotype_id=genotype_id)

        return genotype_id, genotype
