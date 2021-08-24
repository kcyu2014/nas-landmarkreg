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
# import warnings
import logging
import shutil
from abc import ABC, abstractmethod
import utils
import os
import torch
import time
import torch.nn as nn

from collections import OrderedDict
from functools import partial
from operator import itemgetter

import torch.backends.cudnn as cudnn
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats.stats import kendalltau

from torch.utils.tensorboard import SummaryWriter

# after import torch, add this
from nasws.cnn.utils import Rank, compute_percentile, compute_sparse_kendalltau, sort_hash_perfs
import nasws.cnn.search_space as search_space
from nasws.cnn import procedures as procedure_ops
from nasws.cnn.visualization.tool import darts_nds_map_state_dict_keys_from_non_wsbn_to_wsbn
from nasws.cnn.operations.loss import CrossEntropyLabelSmooth
from nasws.dataset import load_supernet_imagenet, load_supernet_cifar10

from visualization.plot_rank_change import process_rank_data_nasbench
from visualization.process_data import *

cudnn.benchmark = True


class CNNSearchPolicy(ABC):
    """
    Search policy for CNN model.
    Base method is Oneshot

    """

    trained_model_spec_ids = []
    
    model_spec_id = None
    model_spec = None
    _change_model_fn = None
    _datasets = None

    @property
    def change_model_spec_fn(self):
        if self._change_model_fn:
            return self._change_model_fn
        else:
            self._change_model_fn = self.search_space.change_model_spec
            return self._change_model_fn

    @property
    def criterion(self):
        return self._loss
    
    @property
    def eval_criterion(self):
        if self._eval_loss:
            return self._eval_loss
        else:
            return self._loss

    @property
    def eval_result(self):
        """ associate with running_stats. to better save this. 
        Format: dict:
                Key: model spec ID or architecture spec
                Val: tuple (acc, obj)
        """
        if 'eval_result' in self.running_stats.keys():
            return self.running_stats['eval_result']
        else:
            self.running_stats['eval_result'] = OrderedDict()
            return self.running_stats['eval_result']

    @property
    def epoch(self):
        if 'epoch' in self.running_stats.keys():
            e = self.running_stats['epoch']
        else:
            e = 0
        self.args.current_epoch = e
        return e
    
    @epoch.setter
    def epoch(self, epoch):
        self.running_stats['epoch']  = epoch
        self.args.current_epoch = epoch

    @property
    def ranking_per_epoch(self):
        if 'ranking_per_epoch' in self.running_stats.keys():
            return self.running_stats['ranking_per_epoch']
        else:
            logging.info("Initializing one")
            self.running_stats['ranking_per_epoch'] = OrderedDict()
            return self.running_stats['ranking_per_epoch']

    @property
    def evaluate_model_spec_ids(self):
        logging.warning("please remove this, use self.search_space.evaluate_model_spec_ids instead.")
        raise ValueError("TODO")
        # return self.search_space.evaluate_model_spec_ids

    def evaluate_model_spec_id_pool(self):
        logging.warning("please remove this, use self.search_space.evaluate_model_spec_id_pool instead.")
        raise ValueError()
        # return self.search_space.evaluate_model_spec_id_pool()

    def __init__(self, args, sub_dir_path=None):
        super(CNNSearchPolicy, self).__init__()
        self.args = args
        self._loss = None
        self._eval_loss = None
    
        if self.args.debug:
            torch.autograd.set_detect_anomaly(True)

        self.initialize_misc()
        # Random seed should be set once the Policy is created.
        logging.info(f"setting random seed as {args.seed}")
        utils.torch_random_seed(args.seed)
        logging.info('gpu number = %d' % args.gpus)
        logging.info("args = %s", args)

        # metrics to track
        # self.ranking_per_epoch = OrderedDict()
        self.search_space = None # store the search space.
        self.model = None # store the model
        self.model_fn = None
        self.amp = None
        self.running_stats = OrderedDict() # store all running status.

        # do it here because, Rank loss needs search space
        self.initialize_search_space()

        # to log the training results.
        self.logging_fn = self.logging_at_epoch
        self.test_fn = procedure_ops.evaluate_procedure.evaluate_normal
        # load the train methods.
        if args.supernet_train_method in ['spos']:
            """ Fundamental baseline training methods
            sample 1 architecture per batch
            train supernet
            Conv op has maximum possible filter channels (== output size of cell)
            Random a chunk of it.
            """
            train_fn = procedure_ops.darts_train_model
            self.train_fn = partial(train_fn, args=self.args, architect=None, sampler=self.random_sampler)
            self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
        elif args.supernet_train_method == 'fairnas':
            """
            Extend darts training method with FairNas strategy. It is not possible to use directly the FairNAS,
            but we can extend it into 2 method.
            """
            train_fn = procedure_ops.fairnas_train_model_v1
            self.train_fn = partial(train_fn, args=self.args, architect=None,
                                    topology_sampler=self.random_sampler,
                                    op_sampler=self.op_sampler
                                    )
            self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
        elif args.supernet_train_method == 'maml':
            """
            Implement the MAML method, to train the supernet for now. It will support other policy later.
            """
            train_fn = procedure_ops.maml_nas_weight_sharing
            self.train_fn = partial(train_fn, args=self.args, architect=None, sampler=self.random_sampler)
            self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
            self.logging_fn = self.logging_maml
            torch.autograd.set_detect_anomaly(True)

        elif args.supernet_train_method == 'spos_rankloss':
            train_fn = procedure_ops.darts_train_model_with_landmark_regularization
            logging.info('Enabling Ranking loss function {} and procedure {}.'.format(
                args.landmark_loss_fn, args.landmark_loss_procedure))
            self.train_fn = partial(train_fn, args=self.args, architect=None, sampler=self.random_sampler,
                                    landmark_loss_step=procedure_ops.landmark_loss_step_fns[args.landmark_loss_procedure],
                                    search_space=self.search_space
                                    )
            self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
            logging.info("Landmark archs selected {}".format(self.search_space.landmark_topologies[0]))
        elif args.supernet_train_method == 'spos_rankloss_maml':
            train_fn = procedure_ops.maml_ranking_loss_procedure
            logging.info('Enabling Ranking loss function {} and procedure {}.'.format(
                args.landmark_loss_fn, args.landmark_loss_procedure))
            self.train_fn = partial(train_fn, args=self.args, architect=None, sampler=self.random_sampler,
                                    landmark_loss_step=procedure_ops.landmark_loss_step_fns[args.landmark_loss_procedure],
                                    search_space=self.search_space
                                    )
            self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)
            logging.info("Landmark archs selected {}".format(self.search_space.landmark_topologies[0]))
            self.logging_fn = self.logging_maml
        else:
            logging.debug(f'{args.supernet_train_method} is not supported in the base class!')
        
    def initialize_misc(self, mode=None):
        """ initialize path and logger """
        args = self.args
        if not args.continue_train:
            self.sub_directory_path = mode or '{}_SEED_{}'.format(args.supernet_train_method,
                                                                          args.seed)
            self.exp_dir = os.path.join(args.main_path, self.sub_directory_path)
            utils.create_exp_dir(self.exp_dir)
            utils.save_json(args, self.exp_dir + '/args.json')
        
        if args.visualize:
            self.viz_dir_path = utils.create_viz_dir(self.exp_dir)

        if args.tensorboard:
            self.tb_dir = self.exp_dir
            tboard_dir = os.path.join(args.tboard_dir, self.sub_directory_path)
            self.writer = SummaryWriter(tboard_dir)

       # Set logger and directory.
        self.logger = utils.get_logger(
            mode or 'train_search',
            file_handler=utils.get_file_handler(os.path.join(self.exp_dir, 'log.txt')),
            level=logging.INFO if not args.debug else logging.DEBUG
        )


    """ Procedure of search """
    # Search happens here.
    def run(self):
        """
        Procedure of training. This run describes the entire training procedure
        Parsed arguments:
            train_fn :      train_queue, valid_queue, model, criterion, optimizer, lr
            valid_model :   model, valid_queue, self.model_spec_id, self.model_spec

        :return:
        """
        model, optimizer, scheduler = self.initialize_model()
        train_queue, valid_queue, test_queue, criterion = self.initialize_run()
        args = self.args
        fitness_dict = {}
        
        logging.info(">> Begin the search with supernet method : {}".format(args.supernet_train_method))
        # start from current epoch.
        if args.debug:
            self.post_search_phase()


        for epoch in range(self.epoch, args.epochs):
            args.current_epoch = epoch
            lr = scheduler.get_last_lr()[0]
            train_acc, train_obj = self.train_fn(train_queue, valid_queue, model, criterion, optimizer, lr)
            self.logging_fn(train_acc, train_obj, epoch, 'Train', display_dict={'lr': lr})
            scheduler.step()

            # validation
            valid_acc, valid_obj = self.validate_model(model, valid_queue, self.model_spec_id, self.model_spec)
            self.logging_fn(valid_acc, valid_obj, epoch, 'Valid')
            self.save_checkpoint(model, optimizer)
            
            # save the imagenet results every 10 epochs ...
            if 'imagenet' in self.args.dataset and epoch % 10 == 0 and epoch > 0:
                # duplicate for imagenet for easy recovery...
                self.save_checkpoint(model, optimizer, epoch)
            
            if not self.check_should_save(epoch):
                continue
            # evaluate process.
            self.save_duplicate_arch_pool('valid', epoch)
            fitness_dict = self.evaluate(epoch, test_queue, fitnesses_dict=fitness_dict, train_queue=train_queue)
            self.save_results(epoch, rank_details=True)

        logging.info(">> Search phase finished! Evaluate here for better results.")
        # add later, return the model specs that is evaluated across the time.
        # Process the ranking in the end, re
        # turn the best of training.
        return self.post_search_phase()

    def validate_model(self, current_model, data_source, current_geno_id, current_genotype, batch_size=10):
        complete_valid_queue = data_source
        _valid_queue = []
        nb_batch_per_model, nb_models, valid_model_pool = self.\
            search_space.validate_model_indices(len(complete_valid_queue))
        total_valid_acc = 0.
        total_valid_obj = 0.

        valid_accs = OrderedDict()
        valid_objs = OrderedDict()

        for step, val_d in enumerate(complete_valid_queue):
            if self.args.debug:
                if step > 100:
                    logging.debug("Break after 100 step in validation step.")
                    break

            _valid_queue.append(val_d)
            if step % nb_batch_per_model == 0 and step > 0:
                _id = valid_model_pool[min(int(step / nb_batch_per_model), nb_models - 1)]
                current_model = self.change_model_spec_fn(current_model, self.search_space.topologies[_id])
                current_model.eval()
                _valid_acc, _valid_obj = self.eval_fn(_valid_queue, current_model, self.eval_criterion)
                # logging.info(f"model id {valid_model_pool[_id]} acc {_valid_acc} loss {_valid_obj}")
                # update the metrics
                total_valid_acc += _valid_acc
                total_valid_obj += _valid_obj
                # store the results
                valid_accs[_id] = _valid_acc
                valid_objs[_id] = _valid_obj
                _valid_queue = []

        self.save_arch_pool_performance(archs=list(valid_accs.keys()), perfs=list(valid_accs.values()), prefix='valid')
        return total_valid_acc / nb_models, total_valid_obj/ nb_models

    def evaluate(self, epoch, data_source, fitnesses_dict=None, train_queue=None, model_spec_id_pool=None):
        """
        Full evaluation of all possible models.
        :param epoch:
        :param data_source:
        :param fitnesses_dict: Store the model_spec_id -> accuracy
        :return: eval_result
            {model_spec_id: (acc, obj)}
                the evaluation results are also returned.
        """
        # Make sure this id pool is not None.
        model_spec_id_pool = model_spec_id_pool or self.search_space.evaluate_model_spec_id_pool()
        if 'rankloss' in self.args.supernet_train_method:
            # add this for evaluate landmark architectures separately.
            landmark_model_id_pool, landmark_specs = self.search_space.landmark_topologies
            landmark_rank_gens, landmark_eval_results = procedure_ops.evaluate_procedure.evaluate_normal(
                self, self.parallel_model, {}, landmark_model_id_pool, data_source, self.change_model_spec_fn,
                self.eval_criterion
            )
            # save the ranking for landmark architectures.
            if 'landmark_ranking_per_epoch' not in self.running_stats.keys():
                self.running_stats['landmark_ranking_per_epoch'] = OrderedDict({epoch: landmark_rank_gens})
            else:
                self.running_stats['landmark_ranking_per_epoch'][epoch] = landmark_rank_gens
            
            logging.info("Rank results for landmark architectures ...")
            self._save_results(
                {'ranking_per_epoch': self.running_stats['landmark_ranking_per_epoch']}, epoch,
                rank_details=True, filename='landmark_rank.json', random_kdt=False, prefix='landmark'
            )
        
        rank_gens, eval_result = procedure_ops.evaluate_procedure.evaluate_normal(
            self, self.parallel_model, fitnesses_dict, model_spec_id_pool, data_source, self.change_model_spec_fn,
            self.eval_criterion
        )

        self.ranking_per_epoch[epoch] = rank_gens
        self.eval_result[epoch] = eval_result

        self.logger.info('VALIDATION RANKING OF PARTICLES')
        for pos, elem in enumerate(rank_gens):
            self.logger.info(f'particle gen id: {elem[1].geno_id}, acc: {elem[1].valid_acc}, obj {elem[1].valid_obj}, '
                             f'hash: {elem[0]}, pos {pos}')

        # save the eval arch pool.
        archs = [elem[1].geno_id for elem in rank_gens]
        perfs = [elem[1].valid_acc for elem in rank_gens]
        self.save_arch_pool_performance(archs, perfs, prefix='eval')
        self.save_duplicate_arch_pool(prefix='eval', epoch=epoch)
        self.search_space.eval_model_spec_id_rank(archs, perfs)
    
        if self.writer:
            # process data into list.
            accs_after, objs_after = zip(*eval_result.values())
            tensorboard_summarize_list(accs_after, writer=self.writer, key='neweval_after/acc', step=epoch, ascending=False)
            tensorboard_summarize_list(objs_after, writer=self.writer, key='neweval_after/obj', step=epoch)

        return eval_result

    def post_search_phase(self):
        """
        Running the policy search on a given shared parameters.
        It should be built from simple use-case, i.e. testing only, to, training after the moment.

        :param model:
        :param optimizer:
        :param train_queue:
        :param valid_queue:
        :param args:
        :param search_space:
        :return: res_dict, best_valids, best_tests from evlutionary search
        """
        # merge the evaluation phase inside.
        # optimizer = self.optimizer
        # scheduler = self.scheduler
        # model = self.parallel_model
        # reset the random seed for reproducibility
        # evaluate before the post-search phase.

        if not self.args.evaluate_after_search:
            # return 0, None
            fitness_dict = self.evaluate(self.epoch, self._datasets[2], train_queue=self._datasets[0])
            self.save_results(self.epoch, rank_details=True)
            ep_k = [k for k in self.ranking_per_epoch.keys()][-1]
            best_id = self.ranking_per_epoch[ep_k][-1][1].geno_id
            return best_id, self.search_space.topologies[best_id]
        
        logging.info(">=== Post Search Phase ====")
        epoch = self.epoch
        self.save_duplicate_arch_pool('valid', epoch)
        tr, va, te = self.load_dataset(shuffle_test=False)

        utils.torch_random_seed(self.args.seed)
        tr, va, te = self.load_dataset(shuffle_test=True)
        nb_batch_per_eval = self.args.evaluate_nb_batch
        if self.args.neweval_num_train_batches > 0:
            query_fn = partial(procedure_ops._query_model_with_train_further, nb_batch=nb_batch_per_eval,
                               model=self.parallel_model,
                               train_queue=tr, valid_queue=va, test_queue=te, policy=self)
            logging.info("> Finetune before evaluation query_fn.")
        else:
            query_fn = partial(procedure_ops._query_model_by_eval,
                               nb_batch=nb_batch_per_eval,
                               model=self.parallel_model,
                               valid_queue=va, test_queue=te, search_space=self.search_space)
            logging.info("> normal evaluation query_fn.")

        logging.info("> Using sampler {}".format(self.args.evaluate_sampler))
        if self.args.evaluate_sampler == 'random':
            res = procedure_ops.run_random_search_over_cnn_search(self.search_space, self.args, query_fn)
        elif self.args.evaluate_sampler == 'evolutionary':
            res = procedure_ops.run_evolutionary_search_on_search_space(self.search_space, self.args, query_fn)
        else:
            raise NotImplementedError(f"self.args.evaluate_sampler {self.args.evaluate_sampler} not yet.")

        search_res_path = self.args.main_path + '/' + self.args.evaluate_sampler + '_results.json'
        search_arch_path = self.args.main_path + '/' + self.args.evaluate_sampler + '_archs.json'
    
        # implement this outside the loop ...
        # test if all is id
        res_dict = res[0]
        res_specs = res[-1]

        mids = list(res_dict.keys())
        est_perfs = [res_dict[i]['test_accuracy'] for i in mids]
        if all(map(lambda x: isinstance(x, int), list(mids))):
            utils.save_json(res, search_res_path)
            # compute the kendall tau and do the best    
            gt_ranks = [self.search_space.rank_by_mid[i] for i in mids]
            logging.info("Kendall tau of given search: {}".format(kendalltau(gt_ranks, est_perfs).correlation))
            if self.args.neweval_num_train_batches > 0:
                est_perfs_old = [res_dict[i]['test_accuracy_before'] for i in mids]
                logging.info("Kendall tau of given search before finetune: {}".format(kendalltau(gt_ranks, est_perfs_old).correlation))

            # post search top 5 results
            top_K_indices = np.argsort(est_perfs)[::-1][:5]
            best_ids = [mids[i] for i in top_K_indices]
            best_specs = [self.search_space.topologies[i] for i in best_ids]
            utils.save_json([best_ids, [str(s) for s in best_specs]], search_arch_path)
            # let pick 5 best models using res.
        else:
            # with new architecture, we just simply return the top 5 by the validation accuracy...
            top_K_indices = np.argsort(est_perfs)[::-1][:5]
            best_ids = [mids[i] if isinstance(mids[i], int) else -1 for i in top_K_indices]
            best_specs = [res_specs[mids[i]] for i in top_K_indices]
            utils.save_json([best_ids, [str(s) for s in best_specs]], search_arch_path)
        return best_ids, best_specs

    ###########################
    ###   Utility functions
    ###########################
    def cpu(self):
        self.model = self.model.cpu()
    
    def initialize_search_space(self):
        """ This function should align with dataset """
        if 'nasbench201' in self.args.search_space:
            # NB201 search space
            self.search_space = search_space.NASBench201SearchSpace(self.args)
            self.model_fn = self.search_space.model_fn
        elif 'darts_nds' in self.args.search_space:
            self.search_space = search_space.DARTSSearchSpaceNDS(self.args)
            self.model_fn = self.search_space.model_fn
        elif 'nasbench101' in self.args.search_space:
            from utils import DictAttr, wrap_config
            # from nasws.cnn.search_space.nasbench101 import nasbench_build_config
            from nasws.cnn.search_space.nasbench101.lib.config import build_config as nasbench_build_config
            nasbench_search_config = DictAttr(nasbench_build_config())
            wrap_config(nasbench_search_config, 'nasbench', args=self.args,
                        keys=[
                            'module_vertices',
                            # add more if necessary.
                        ])
            self.args.nasbench_config = nasbench_search_config
            if 'fixchannel' in self.args.search_space:
                self.search_space = search_space.NasBenchSearchSpaceFixChannels(self.args)
            else:
                self.search_space = search_space.NASbenchSearchSpace(self.args)
            self.model_fn = self.search_space.model_fn
        else:
            raise NotImplementedError(f'initialize search space not supported for {self.args.search_space}')
        
        self._change_model_fn = self.search_space.change_model_spec

    def initialize_run(self, sub_dir_path=None):
        tr, va, te = self.load_dataset()
        logging.info('Creating the loss function with CrossEntropy')
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        if self.args.label_smooth > 0:
            self._eval_loss = criterion
            logging.info(f'Label smoothing enabled with {self.args.label_smooth}')
            criterion = CrossEntropyLabelSmooth(self.num_classes, self.args.label_smooth)
            criterion = criterion.cuda()
        self._loss = criterion
        
        return tr, va, te, criterion

    def initialize_model(self, resume=True):
        """
        Initialize model, may change across different model.
        :return:
        """
        args = self.args
        model = self.model_fn(args)
        num_gpus = torch.cuda.device_count()

        if args.apex_enable:
            args.distributed = False
            if 'WORLD_SIZE' in os.environ:
                args.distributed = int(os.environ['WORLD_SIZE']) > 1
            args.gpu = num_gpus
            args.world_size = 1
            args.learning_rate = args.learning_rate*float(args.batch_size*args.world_size)/1024
            model = model.cuda().to(memory_format=torch.contiguous_format)
            
        else:
            if num_gpus > 0:
                model = model.cuda()

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
            logging.info("Creating SGD : init_lr {} weight_decay {}, momentum {}".format(
                args.learning_rate, args.momentum, args.weight_decay))
        elif args.optimizer == 'rmsprop':
            from nasws.cnn.search_space.nasbench101.optimizer import RMSprop as RMSpropTF
            optimizer = RMSpropTF(
                model.parameters(),
                args.learning_rate,
                eps=1.0,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )
            logging.info("Creating RMSProp : init_lr {} weight_decay {}, momentum {}".format(
                args.learning_rate, args.momentum, args.weight_decay))

        elif args.optimizer == 'adam':
            raise ValueError("todo later")
        else:
            raise NotImplementedError('optimizer not supported...')
        
        if args.apex_enable: 
            import apex.amp as amp
            model, optimizer = amp.initialize(
                model, optimizer,
                opt_level='O1', # official mixed precision
                # keep_batchnorm_fp32=True, # bn 32 to accelarate further.
                loss_scale=None)    # do not scale
            args.apex_opt_level='O1'
            self.amp = amp
        
        self.model = model # this is a normal cpu model...
        self.parallel_model = nn.DataParallel(model) if num_gpus > 1 else model

        lr_epochs = args.epochs if args.epochs_lr <= 0 else args.epochs_lr
        logging.info(f'Creating learning rate scheduler: {args.learning_rate_scheduler} with max epoch {lr_epochs}')
        if args.learning_rate_scheduler == 'cosine':
            # scheduler as Cosine.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(lr_epochs), eta_min=args.learning_rate_min)
        elif args.learning_rate_scheduler == 'cosinewarm':
            # scheduler as Cosine With Warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, int(lr_epochs / 2), eta_min=args.learning_rate_min
            )
        elif args.learning_rate_scheduler == 'cosineimagenet':
            from nasws.cnn.operations.lr_scheduler import CosineWarmupImagenetScheduler
            scheduler = CosineWarmupImagenetScheduler(
                optimizer, args.supernet_warmup_epoch, float(lr_epochs), eta_min=args.learning_rate_min
            )
        elif args.learning_rate_scheduler == 'step':
            # step wise lr setting
            # scheduler = torch.optim.lr_scheduler.StepLR(
            #     optimizer, step_size=int(args.epochs * 0.6 / 3), gamma=0.1, last_epoch=int(args.epochs * 0.7)
            # )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(lr_epochs * i) for i in [0.3, 0.6, 1.0]], gamma=0.1)
        elif args.learning_rate_scheduler == 'constant':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=1.0)
        else:
            raise ValueError("LR Scheduler {} not supported.".format(args.learning_rate_scheduler))
    
        self.optimizer = optimizer
        self.scheduler = scheduler
        # here it is, but lets try to reload this.
        if args.resume and resume:
            self.resume_from_checkpoint()

        return self.parallel_model, optimizer, scheduler

    def load_dataset(self, shuffle_test=False):
        if self._datasets:
            return self._datasets
        
        args = self.args
        if args.dataset == 'cifar10':
            train_queue, valid_queue, test_queue = load_supernet_cifar10(args, shuffle_test)
            self.num_classes = 10
        elif args.dataset == 'imagenet':
            # train_queue, valid_queue, test_queue = load_supernet_cifar10(args, shuffle_test, debug_imgnet=True)
            train_queue, valid_queue, test_queue = load_supernet_imagenet(args)
            self.num_classes = 1000
        else:
            raise NotImplementedError("Temporary not used.")
        self._datasets = train_queue, valid_queue, test_queue
        return self._datasets

    @staticmethod
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

    def op_sampler(self, model, architect, args):
        return self.search_space.op_sampler(model, architect, args)

    def random_sampler(self, model, architect, args):
        return self.search_space.random_sampler(model, architect, args)
    
    @staticmethod
    def _compute_kendall_tau(ranking_per_epoch, compute_across_time=False):
        """
        Compute Kendall tau given the ranking per epochs.

        :param ranking_per_epoch:
        :param compute_across_time: True for ranking-per-epoch always fixed, False for dynamic list of models.
        :return: kd_tau dict{epoch_key: KendallTau}
        """
        if compute_across_time:
            # Compute Kendall tau for every epochs and save them into result.
            epoch_keys = [k for k in reversed(ranking_per_epoch.keys())]
            epoch_keys.insert(0, 10000000)
            kd_tau = {}
            for ind, k in enumerate(epoch_keys[:-1]):
                elem = []
                if ind == 0:
                    # Sort the ground-truth ranking
                    p = sorted([elem[1] for elem in ranking_per_epoch[epoch_keys[ind + 1]]], key=itemgetter(3))
                    rank_1 = np.array([elem.geno_id for elem in p], dtype=np.uint)
                else:
                    rank_1 = np.array([elem[1].geno_id for elem in ranking_per_epoch[k]], dtype=np.uint)
                for j in epoch_keys[ind + 1:]:
                    rank_2 = np.array([elem[1].geno_id for elem in ranking_per_epoch[j]], dtype=np.uint)
                    elem.append(kendalltau(rank_1, rank_2))
                kd_tau[k] = elem
            logging.info("Latest Kendall Tau (ground-truth vs {}): {}".format(epoch_keys[1], kd_tau[10000000][0]))
            return kd_tau, kd_tau[10000000][0].correlation
        else:
            # Dynamic ranking per epoch size, thus only compute the KDT against the final ranking.
            epoch_keys = [k for k in reversed(ranking_per_epoch.keys())]
            kd_tau = {}
            # only sort across the ground-truth
            for ind, k in enumerate(epoch_keys):
                p = sorted([elem[1] for elem in ranking_per_epoch[k]], key=itemgetter(3))
                rank_gt = np.array([elem.geno_id for elem in p], dtype=np.uint)
                rank_2 = np.array([elem[1].geno_id for elem in ranking_per_epoch[k]], dtype=np.uint)
                kd_tau[k] = kendalltau(rank_gt, rank_2)

            # IPython.embed(header='check kendall tau computation again for DARTS search space')
            kd_tau[10000000] = kd_tau[epoch_keys[0]]
            logging.info("Latest Kendall Tau (ground-truth vs {}): {}".format(epoch_keys[0], kd_tau[epoch_keys[0]][0]))
            return kd_tau, kd_tau[epoch_keys[0]][0]

    def _save_ranking_results(self, save_data, epoch,
                              prefix=None,
                              compute_kdt_before=False,
                              sparse_kdt=True, sparse_kdt_threshold=1e-3,
                              percentile=True, percentile_top_K=(3, 5, 10, 20),
                              random_kdt=True, random_kdt_numrepeat=5, random_kdt_num_archs=(10, 20, 50, 100)):
        """
        Save the ranking results if necessary.

        13.09.2019: Adding the sparse kendall tau, percentile, random kendall tau.

        :param save_data:
        :param epoch:
        :param prefix: Prefix to the tensorboard scalar.
        :param compute_kdt_before: True to compute the kendall tau for additional evaluation approachs.
        :param sparse_kdt: True to compute the sparse kendall tau based on the GT accuracy.
        :param percentile: True to compute the top K model percentile.
        :param percentile_top_K: Number of top K architectures for percentile.
        :param random_kdt: True to compute the random K architectures's kendall tau.
        :param random_kdt_numrepeat: Number of repeated times for this random kdt
        :param random_kdt_num_archs: Number of architectures for random kdt
        :return: None
        """
        prefix_name = lambda prefix, name: name if prefix is None else f'{prefix}-{name}'

        try:
            fname = prefix_name(prefix, f'rank_change-{epoch}.pdf')
            fig = process_rank_data_nasbench(save_data, os.path.join(self.exp_dir, fname))
            if self.writer:
                self.writer.add_figure(tag=fname.split('.')[0].replace('_','-'), figure=fig, global_step=epoch)
        except Exception as e:
            logging.warning(e)

        try:
            ranking_per_epoch = save_data['ranking_per_epoch']
        except KeyError as e:
            logging.warning("save_data parsed into _save_ranking_results expect having key ranking_per_epoch"
                            "; got {}. Using self.ranking_per_epoch instead".format(save_data.keys()))
            ranking_per_epoch = self.ranking_per_epoch

        # Compute Kendall tau for every epochs and save them into result.
        # IPython.embed()
        kd_tau, kd_tau_report = self._compute_kendall_tau(ranking_per_epoch)
        save_data['kendaltau'] = kd_tau

        if compute_kdt_before and 'ranking_per_epoch_before' in self.running_stats.keys():
            kd_tau_before, kd_tau_before_report = self._compute_kendall_tau(
                self.running_stats['ranking_per_epoch_before'])
            save_data['kendaltau_before'] = kd_tau_before

        if self.writer is not None:
            p = sorted([elem[1] for elem in ranking_per_epoch[epoch]], key=itemgetter(2))
            tensorboard_summarize_list(
                [e[0] for e in p], self.writer, prefix_name(prefix, 'eval_acc'), epoch, ascending=False
            )
            tensorboard_summarize_list(
                [e[1] for e in p], self.writer, prefix_name(prefix, 'eval_obj'), epoch, ascending=True
            )
            self.writer.add_scalar(prefix_name(prefix, 'eval_kendall_tau'), kd_tau_report, epoch)
            if compute_kdt_before and 'ranking_per_epoch_before' in self.running_stats.keys():
                self.writer.add_scalar(prefix_name(prefix, 'eval_kendall_tau/original'), kd_tau_before_report, epoch)

        # add these and collect writer keys
        if any([sparse_kdt, percentile, random_kdt]):

            data = ranking_per_epoch[epoch]
            # ranking by valid accuracy
            model_ids = [elem[1][3] for elem in data]
            model_perfs = [elem[1][0] for elem in data]
            model_ids, model_perfs = sort_hash_perfs(model_ids, model_perfs)
            model_gt_perfs = self.search_space.query_gt_perfs(model_ids)
            sorted_indices = np.argsort(model_perfs)[::-1]
            sorted_model_ids = [model_ids[i] for i in sorted_indices]

            # IPython.embed(header='checking the saving here.')
            add_metrics = {}
            if sparse_kdt:
                if not isinstance(sparse_kdt_threshold, (tuple, list)):
                    sparse_kdt_threshold = [sparse_kdt_threshold]
                for th in sparse_kdt_threshold:

                    kdt = compute_sparse_kendalltau(model_ids, model_perfs, model_gt_perfs,
                                                                   threshold=th)
                    add_metrics[prefix_name(prefix, f'eval_kendall_tau/sparse_{th}')] = kdt.correlation

            if percentile:
                for top_k in percentile_top_K:
                    res = compute_percentile(sorted_model_ids,
                                             self.search_space.num_architectures,
                                             top_k,
                                             verbose=self.args.debug)
                    mname = prefix_name(prefix, 'percentile')
                    add_metrics[f'{mname}/min_{top_k}'] = res.min()
                    add_metrics[f'{mname}/median_{top_k}'] = np.median(res)
                    add_metrics[f'{mname}/max_{top_k}'] = res.max()
                logging.info("{} of top {}: {} - {} - {}".format(
                    mname,
                    top_k, res.min(), np.median(res), res.max()))

            if random_kdt:
                for subsample in random_kdt_num_archs:
                    if subsample > len(sorted_model_ids):
                        continue
                    kdt_final = []
                    for _ in range(random_kdt_numrepeat):
                        sub_model_indices = sorted(
                            np.random.choice(np.arange(0, len(sorted_model_ids)), subsample, replace=False).tolist())
                        sub_model_ids = [sorted_model_ids[i] for i in sub_model_indices]
                        kdt = kendalltau(sub_model_ids, list(reversed(sorted(sub_model_ids))))
                        kdt_final.append(kdt.correlation)

                    kdt_final = np.asanyarray(kdt_final, dtype=np.float)
                    mname = prefix_name(prefix, 'eval_kendall_tau')
                    add_metrics[f'{mname}/random_{subsample}_min'] = kdt_final.min()
                    add_metrics[f'{mname}/random_{subsample}_max'] = kdt_final.max()
                    add_metrics[f'{mname}/random_{subsample}_mean'] = kdt_final.mean()

                    logging.info("Random subsample {} archs: kendall tau {} ({},{})".format(
                        subsample, kdt_final.mean(), kdt_final.min(), kdt_final.max()))

            # end of additioanl metrics
            if self.writer:
                for k, v in add_metrics.items():
                    self.writer.add_scalar(k, v, epoch)

        return save_data

    def _save_results(self, save_data, epoch, rank_details=False, filename='result.json', **kwargs):
        if rank_details:
            save_data = self._save_ranking_results(save_data, epoch, **kwargs)

        utils.save_json(save_data, os.path.join(self.exp_dir, filename))
        # if hasattr(self, 'writer') and self.writer is not None:
        #     self.writer.export_scalars_to_json(os.path.join(self.exp_dir, 'tb_scalars.json'))
    
    def save_results(self, epoch, rank_details=True):
        save_data = {
            'ranking_per_epoch': self.ranking_per_epoch,
            'trained_model_spec_per_steps': self.trained_model_spec_ids,
        }
        # for other to overwrite.
        return self._save_results(save_data, epoch, rank_details, sparse_kdt=True, percentile=True, random_kdt=True)

    def save_policy_results(self, epoch, sampled_arch_ids, sample_perfs=None):
        """Save policy results, used for policy such as DARTS, NAO and ENAS, to track the intermediate results

        Parameters
        ----------
        epoch : int
            epoch number
        sampled_arch_ids : list
            List of sampled architecture IDs
        """
        # Make sure this id pool is not None.
        model_spec_id_pool = sampled_arch_ids
        
        self.logger.info(f'saving policy sampled ID at epoch {epoch}')
        # save the eval arch pool.
        archs = []
        for i in model_spec_id_pool:
            if isinstance(i, int):
                archs.append(self.search_space.topology_by_id(i))
            else:
                archs.append(i)

        perfs = sample_perfs if sample_perfs else [0] * len(archs)
        
        for i, pos in enumerate(model_spec_id_pool):
            self.logger.info(f'particle gen id: {i}, performance: {perfs[i]}'
                             f'spec: {archs[i]}, pos {pos}')

        self.save_arch_pool_performance(model_spec_id_pool, perfs, prefix='sampler')
        self.save_duplicate_arch_pool(prefix='sampler', epoch=epoch)
    
        if self.writer:
            # process data into list
            gt_perfs = self.search_space.query_gt_perfs(sampled_arch_ids)
            if gt_perfs:
                tensorboard_summarize_list(np.array(gt_perfs), writer=self.writer, key='policy/gt_acc', step=epoch, ascending=False)
                percentile = np.array(sampled_arch_ids) / self.search_space.num_architectures
                self.writer.add_scalar('policy/percentile_median', np.median(percentile), epoch)
                self.writer.add_scalar('policy/percentile_max', np.max(percentile), epoch)

    def check_should_save(self, epoch):
        """
        invoke the evaluate step, this is also used to update epoch information.
        :param epoch:
        :return:
        """
        self.running_stats['epoch'] = epoch
        self.epoch
        if self.args.extensive_save and epoch > 50:
            return any([(epoch - i) % self.args.save_every_epoch == 0 for i in range(3)])
        
        return epoch % self.args.save_every_epoch == 0 and epoch > 0

    # logging for normal one.
    def logging_at_epoch(self, acc, obj, epoch, keyword, display_dict=None):
        message = f'{keyword} at epoch {epoch} | loss: {obj:8.2f} | top_1_acc: {acc:8.2f}'
        if display_dict:
            for k, v in display_dict.items():
                message += f' | {k}: {v} '

        logging.info(message)
        self.running_stats['epoch'] = epoch
        if self.writer:
            self.writer.add_scalar(f'{keyword}/loss', obj, epoch)
            self.writer.add_scalar(f'{keyword}/top_1_acc', acc, epoch)
            if display_dict:
                for k, v in display_dict.items():
                    self.writer.add_scalar(f'{keyword}/{k}', v, epoch)

    def logging_maml(self, acc, obj, epoch, keyword, **kwargs):
        """ specifically for MAML procedures"""
        if isinstance(acc, tuple) and len(acc) == 2:
            self.logging_at_epoch(acc[0], obj[0], epoch, keyword + '_task', **kwargs)
            self.logging_at_epoch(acc[1], obj[1], epoch, keyword + '_meta', **kwargs)
        else:
            return self.logging_at_epoch(acc, obj, epoch, keyword, **kwargs)

    def save_checkpoint(self, model, optimizer, backup_epoch=None, other_dict=None):
        d = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'misc': self.running_stats
        }
        if self.amp:
            d['amp'] = self.amp.state_dict()
        if self.scheduler:
            d['scheduler'] = self.scheduler.state_dict()
        if other_dict:
            d.update(other_dict)
        
        utils.save_checkpoint_v2(d, self.exp_dir + '/checkpoint.pt', backup_weights_epoch=backup_epoch)

    def resume_from_checkpoint(self, path=None, epoch=None):
        """ resume the training and restoring the statistics. """
        path = path or self.exp_dir
        if self.args.resume_path and os.path.exists(self.args.resume_path):
            path = self.args.resume_path
        
        if os.path.exists(os.path.join(path, 'checkpoint.pt')):
            res_dict = torch.load(os.path.join(path, 'checkpoint.pt'))
        elif os.path.exists(path) and '.pt' in path[-10:]:
            res_dict = torch.load(path)
        else:
            try:
                res_dict = utils.load_checkpoint_v2(path, epoch=epoch)
            except FileNotFoundError:
                return None
        
        if 'darts_nds' in path:
            # wrapping the keys to counter the changes.
            res_dict['model'] = darts_nds_map_state_dict_keys_from_non_wsbn_to_wsbn(res_dict['model'])
        
        # reload the running status.
        self.running_stats = res_dict['misc']
        self.running_stats['epoch'] += 1

        logging.info("=" * 80)
        logging.info(f"Resume training from epoch {self.epoch}... ")
        # logging.info(f"Reload to start from epoch {self.epoch}")
        if res_dict:
            if hasattr(self.parallel_model, 'module'):
                if not any([k.startswith('module') for k in res_dict['model'].keys()]):
                    mk, uk = self.parallel_model.module.load_state_dict(res_dict['model'], strict=False)
                else:
                    mk, uk = self.parallel_model.load_state_dict(res_dict['model'], strict=False)
                
            else:
                mk, uk = self.parallel_model.load_state_dict(res_dict['model'], strict=False)
            logging.info('model resumed...')
            if len(mk) > 0 and not self.args.debug:
                import warnings
                warnings.warn("Loading model state dicts error: missing keys {}".format(mk))
            self.optimizer.load_state_dict(res_dict['optimizer'])
            logging.info(f'optimizer resumed...')
            if 'scheduler' in res_dict.keys():
                self.scheduler.load_state_dict(res_dict['scheduler'])
                logging.info(f'LR scheduler resumed, lr={self.scheduler.get_last_lr()[0]}')
            else:
                # step to the epoch
                logging.info(f'LR scheduler resume to epoch number {self.epoch}')
                self.scheduler.step(self.epoch)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                self.scheduler.T_max = max(self.args.epochs_lr, self.args.epochs)
                self.scheduler.eta_min = self.args.learning_rate_min
                logging.info(self.scheduler.__dict__)
            else:
                # import ipdb; ipdb.set_trace()
                # raise NotImplementedError(f"TO support correct resume by override T_max. {self.scheduler}")
                logging.warn(f'Do not set the T_max for learning rate scheduler {self.scheduler}')

            if 'amp' in res_dict.keys():
                self.amp.load_state_dict(res_dict['amp'])
                logging.info(f'amp resume')
        else:
            logging.info("No model file found here. start from scratch from epoch {}".format(self.epoch))
        logging.info("=" * 80)

        # load eval results.
        result_path = os.path.join(path, 'result.json')
        if os.path.exists(result_path):
            save_data = utils.load_json(result_path)
            logging.info(f'loading results {save_data.keys()}.')
            for k, v in save_data.items():
                self.running_stats[k] = v
            # process named tuple.
            logging.info("resume the Rank namedtuple")
            rp_dict = self.ranking_per_epoch
            self.running_stats['ranking_per_epoch'] = OrderedDict()
            for k, v in rp_dict.items():
                self.ranking_per_epoch[int(k)] = [[i1, Rank(*i2)] for i1, i2 in v]

        return res_dict

    def save_duplicate_arch_pool(self, prefix, epoch):
        f_pool = os.path.join(self.exp_dir, f'{prefix}_arch_pool')
        f_perf = os.path.join(self.exp_dir, f'{prefix}_arch_pool.perf')
        if os.path.exists(f_pool):
            shutil.copy(f_pool, f_pool + '.{}'.format(epoch))
        if os.path.exists(f_perf):
            shutil.copy(f_perf, f_perf + '.{}'.format(epoch))

    def save_arch_pool_performance(self, archs, perfs, prefix='valid'):
        old_archs_sorted_indices = np.argsort(perfs)[::-1]
        old_archs = [archs[i] for i in old_archs_sorted_indices]
        old_archs_perf = [perfs[i] for i in old_archs_sorted_indices]
        with open(os.path.join(self.exp_dir, f'{prefix}_arch_pool'), 'w') as fa_latest:
            with open(os.path.join(self.exp_dir, f'{prefix}_arch_pool.perf'), 'w') as fp_latest:
                for arch_id, perf in zip(old_archs, old_archs_perf):
                    if isinstance(arch_id, int):
                        arch = self.search_space.process_archname_by_id(arch_id)
                    else:
                        arch = arch_id
                    fa_latest.write('{}\n'.format(arch))
                    fp_latest.write('{}\n'.format(perf))



class CNNWarmupSearchPolicy(CNNSearchPolicy):
    
    # def sample_topologies_by_distance(self, num_architectures):
    #     num_root = num_architectures // 10
    #     num_arch_per_root = 10 - 1
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
        args.evaluate_after_search, ev = False, args.evaluate_after_search
        # warm up
        total_epochs, args.epochs = args.epochs, args.supernet_warmup_epoch
        self.initialize_misc()
        # args.epochs = 100
        # args.supernet_warmup_epoch = 100
        super().run()
        pretrain_checkpoint = self.exp_dir
        args.evaluate_after_search = ev
        # keeps a separate and same place to start the training...
        args.epochs = total_epochs

        args.resume_path = pretrain_checkpoint
        for run_id in range(1):
            # loading the super-net to the checkpoint at warmup...
            # initialize a new folder
            logging.info(f'Run ID {run_id}')
            self.initialize_misc(f'{args.supernet_train_method}-run{run_id}')
            # train as usual...
            super().run()
        
