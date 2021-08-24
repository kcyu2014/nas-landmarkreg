#  ========================================================
#  CONFIDENTIAL - Under development
#  ========================================================
#  Author: Kaicheng Yu with email kaicheng.yu@epfl.ch
#  All Rights Reserved.
#  Last modified: 2019/11/27 下午12:05
#  NOTICE:  All information contained herein is, and remains
#   the property of Kaicheng Yu, if any.  The intellectual and
#   technical concepts contained herein are proprietary to him
#   and his suppliers and may be covered by U.S. and Foreign Patents,
#   patents in process, and are protected by trade secret or copyright law.
#   Dissemination of this information or reproduction of this material
#   is strictly forbidden unless prior written permission is obtained
#   from Kaicheng Yu.
#  ========================================================
import logging
import os
import numpy as np
import utils
from torch.utils.tensorboard import SummaryWriter

from ..cnn_general_search_policies import CNNSearchPolicy
from nasws.cnn.search_space.nasbench201.nasbench201_search_space import NASBench201SearchSpace, CellStructure
from nasws.cnn.search_space.nasbench201.model_search import NASBench201NetSearch, build_nasbench201_search_model
from nasws.cnn.policy.nao_policy.utils_for_nasbench201 import NAOParsingNASBench201
from nasws.cnn.policy.nao_policy.utils import hamming_distance

# # basically isolate the search policy with the search space...
# class NASbench201WeightSharingPolicy(CNNSearchPolicy): 

#     def __init__(self, args):
#         super(NASbench201WeightSharingPolicy, self).__init__(args, sub_dir_path=None)
#     #     self._change_model_fn = NASBench201SearchSpace.change_model_spec

#     # def initialize_search_space(self):
#     #     """ Initialize NASBench102 search space """
#     #     if selfmodel_fn.args.search_space == 'nasbench201':
#     #         # self.model_fn = build_nasbench201_search_model
#     #         self.search_space = NASBench201SearchSpace(self.args)
#     #     elif self.args.search_space == 'nasbench201_subsample':
#     #         # self.model_fn = build_nasbench201_search_model
#     #         self.search_space = NASBench201SearchSpace(self.args)
#     #     else:
#     #         raise NotImplementedError("Not yet support other format.")
#         self.model_fn = self.search_space.


class NASBench201HammingDistancePolicy(CNNSearchPolicy):
    
    def sample_topologies_by_distance(self, num_architectures, iteration=None):
        if iteration is not None:
            num_root = iteration
            num_arch_per_root = 9
            num_architectures = iteration * 10
        else:
            num_root = 1 
            num_arch_per_root = num_architectures - 1

        ids = []
        distance = self.args.landmark_sample_distance
        for _ in range(num_root):
            mid, spec = self.search_space.random_topology()
            arch = NAOParsingNASBench201.parse_model_spec_to_arch(spec)
            arch_set = [arch]
            ids.append(mid)
            for _ in range(num_arch_per_root):
                dist, counter = 0, 0
                n_spec = spec
                nid = None
                while dist <= distance and counter < 500:
                    counter += 1
                    nid, n_spec = self.search_space.mutate_topology(n_spec)
                    n_arch = NAOParsingNASBench201.parse_model_spec_to_arch(n_spec)
                    dist = hamming_distance([n_arch,] * len(arch_set), arch_set)
                if nid:
                    logging.debug(f'sample architecture distance {dist}: ({nid}) {n_spec}')
                    ids.append(nid)
        logging.debug(f'Sampling landmark by distance: {ids}')
        return list(sorted(ids))

    # def sample_topologies_by_distance(self, num_architectures):
    #     num_root = 1
    #     num_arch_per_root = num_architectures - 1
    #     ids = []
    #     distance = self.args.landmark_sample_distance
    #     for _ in range(num_root):
    #         mid, spec = self.search_space.random_topology()
    #         arch = NAOParsingNASBench201.parse_model_spec_to_arch(spec)
    #         ids.append(mid)
    #         for _ in range(num_arch_per_root):
    #             dist, counter = 0, 0
    #             n_spec = spec
    #             nid = None
    #             while dist <= distance and counter < 50:
    #                 counter += 1
    #                 nid, n_spec = self.search_space.mutate_topology(n_spec)
    #                 n_arch = NAOParsingNASBench201.parse_model_spec_to_arch(n_spec)
    #                 dist = hamming_distance([n_arch], [arch])
    #             if nid:
    #                 logging.debug(f'sample architecture distance {dist}: ({nid}) {n_spec}')
    #                 ids.append(nid)

    #     logging.debug(f'Sampling landmark by distance: {ids}')
    #     return list(sorted(ids))

    def initialize_misc(self, mode='warmup'):
        # assert mode in ['warmup', 'landmark']
        args = self.args
        # initialize path and logger
        if not args.continue_train:
            self.sub_directory_path = mode or 'warmup'
            self.exp_dir = os.path.join(self.args.main_path, self.sub_directory_path)
            utils.create_exp_dir(self.exp_dir)
            utils.save_json(args, self.exp_dir + '/args.json')
        
        if self.args.visualize:
            self.viz_dir_path = utils.create_viz_dir(self.exp_dir)

        if self.args.tensorboard:
            self.tb_dir = self.exp_dir
            tboard_dir = os.path.join(self.args.tboard_dir, self.sub_directory_path)
            self.writer = SummaryWriter(tboard_dir)

       # Set logger and directory.
        self.logger = utils.get_logger(
            "train_search",
            file_handler=utils.get_file_handler(os.path.join(self.exp_dir, 'log.txt')),
            level=logging.INFO if not args.debug else logging.DEBUG
        )

    def run(self):
        """
        Procedure of training. This run describes the entire training procedure
        Parsed arguments:
            train_fn :      train_queue, valid_queue, model, criterion, optimizer, lr
            valid_model :   model, valid_queue, self.model_spec_id, self.model_spec

        :return:
        """
        args = self.args
        args.evaluate_after_search = False
        # warm up
        total_epochs, args.epochs = args.epochs, args.supernet_warmup_epoch
        args.epochs_lr = total_epochs
        self.initialize_misc()
        # args.epochs = 100
        # args.supernet_warmup_epoch = 100
        super().run()
        pretrain_checkpoint = self.exp_dir
        # do not decay the learning rate after this point for this experiment please, 
        # otherwise the training will be destroied
        args.learning_rate = args.learning_rate_min
        args.resume_path = pretrain_checkpoint
        args.learning_rate_scheduler = 'cosine'
        args.learning_rate_min = 0
        args.epochs = total_epochs
        args.save_every_epoch = 10 # excessive saving to see the effect (or more data points...)
        for run_id in range(4, 7):
            if args.search_policy == 'hamming':
                self.initialize_misc(f'distance{args.landmark_sample_distance}-run{run_id}')
                # loading the super-net to the checkpoint at warmup...
                self.search_space.landmark_topologies = self.sample_topologies_by_distance(args.landmark_num_archs)    
                logging.info(f">>> Begin the search with one set of landmark sets : {self.search_space._landmark_ids}")
                res = super().run()
                # all initialization will be redo... so please do not 
            elif args.search_policy == 'hamming-iteration':
                self.initialize_misc(f'iteration{args.landmark_num_archs}-run{run_id}')
                self.search_space.landmark_topologies = self.sample_topologies_by_distance(0, iteration=args.landmark_num_archs)
                logging.info(f">>> Begin the search with one set of landmark sets : {self.search_space._landmark_ids}")
                res = super().run()
        return res
    
    def post_search_phase(self):
        return 0, None
