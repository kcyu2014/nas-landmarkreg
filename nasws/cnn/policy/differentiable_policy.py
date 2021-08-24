"""PCDarts result

"""
import os
import logging
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from functools import partial
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

import utils
from .cnn_general_search_policies import CNNSearchPolicy
import nasws.cnn.procedures as procedure_ops
from nasws.cnn.search_space.nasbench101.model_search import NasBenchNetSearchDARTS
from nasws.cnn.search_space.nasbench201.model_search import NASBench201NetSearchDARTS, build_nasbench201_search_model
from nasws.cnn.search_space.darts.model_search import DartsNetworkCIFARSearchDARTS
from nasws.cnn.search_space.darts.model_search_imagenet import DartsNetworkImageNetSearchDARTS


class DifferentiableCNNPolicy(CNNSearchPolicy):
    top_K_complete_evaluate = 200
    top_K_num_sample = 1000
    policy_epochs = 50

    def __init__(self, args) -> None:
        super(DifferentiableCNNPolicy, self).__init__(args)
        # Architect during DARTS training
        self.architect = None

    def initialize_search_space(self):
        # override the model_fn here...
        super().initialize_search_space()
        search_space_name = self.args.search_space
        if 'nasbench101' in search_space_name:
            self.model_fn = NasBenchNetSearchDARTS
        elif 'nasbench201' in search_space_name:
            self.model_fn = build_nasbench201_search_model
        elif 'darts_nds' in search_space_name:
            if self.args.dataset == 'imagenet':
                self.model_fn = DartsNetworkImageNetSearchDARTS
            elif self.args.dataset == 'cifar10':
                self.model_fn = DartsNetworkCIFARSearchDARTS
        else:
            raise ValueError(f'Search space not supported: {search_space_name}')

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
    
    def initialize_differentiable_policy(self, criterion):
        # build the architecture forward for Archiect here.
        def architect_forward_fn(model, inputs, target):
            loss, _, _ = self.search_space.module_forward_fn(model, inputs, target, criterion)
            return loss

        self.architect.module_forward_fn = architect_forward_fn
        if self.args.supernet_train_method == 'darts':
            self.train_fn = partial(
                procedure_ops.pcdarts_train_procedure, args=self.args, sampler=self.random_sampler
                )
        elif self.args.supernet_train_method == 'darts_rankloss':
            self.train_fn = partial(
                procedure_ops.darts_train_model_with_landmark_regularization,
                args=self.args,
                search_space=self.search_space, sampler=self.random_sampler,
                landmark_loss_step=procedure_ops.landmark_loss_step_fns[self.args.landmark_loss_procedure],
            )
        else:
            raise NotImplementedError(f'Method not supported {self.args.supernet_train_method}')

        self.eval_fn = partial(procedure_ops.darts_model_validation, args=self.args)

    def run(self):
        # process UTILS and args
        args = self.args

        #################################
        # warmup here...
        #################################

        # use the pre-train logic, isolate the search and the final phase...
        # pretraining is until args.epochs is done. 
        self.initialize_misc(f'warmup-{args.seed}')
        
        # revise the differentiable policy learning rate settings.
        args.epochs_lr = max(args.epochs_lr, args.epochs + self.policy_epochs)
        
        train_queue, valid_queue, test_queue, train_criterion = self.initialize_run()
        eval_criterion = train_criterion
        parallel_model, optimizer, scheduler = self.initialize_model(resume=False)
        self.initialize_differentiable_policy(train_criterion)

        # redo the resume logic...
        args.resume_path, resume_path = None, args.resume_path
        # try to resume from local checkpoint first...
        if self.resume_from_checkpoint():
            pass
            logging.info('Resume from local checkpoint...')
        else:
            logging.info('Try resume from args.resume_path')
            args.resume_path = resume_path
            self.resume_from_checkpoint()

        self.optimizer = optimizer
        self.scheduler = scheduler
        # Train child model
        # Begin the full loop
        logging.info(">> Warmup with supernet method : {}".format(args.supernet_train_method))
        # start from current epoch.
        for epoch in range(self.epoch, args.epochs):
            args.current_epoch = epoch
            lr = scheduler.get_lr()[0]
            scheduler.step()
            train_acc, train_obj = self.train_fn(
                train_queue, valid_queue, parallel_model, train_criterion, optimizer, lr)
            self.logging_fn(train_acc, train_obj, epoch, 'Train', display_dict={'lr': lr})

            # validation
            valid_acc, valid_obj = self.validate_model(parallel_model, valid_queue, self.model_spec_id, self.model_spec)
            self.logging_fn(valid_acc, valid_obj, epoch, 'Valid')
            self.save_checkpoint(parallel_model, optimizer)
            
            # save the imagenet results every 10 epochs ...
            if 'imagenet' in self.args.dataset and epoch % 10 == 0 and epoch > 0:
                # duplicate for imagenet for easy recovery...
                self.save_checkpoint(parallel_model, optimizer, epoch)
            
            if not self.check_should_save(epoch):
                continue
            # evaluate process.
            self.save_duplicate_arch_pool('valid', epoch)
            self.evaluate(epoch, test_queue, fitnesses_dict=None, train_queue=train_queue)
            self.save_results(epoch, rank_details=True)
        pretrain_checkpoint = self.exp_dir

        #################################
        # train architect weights 
        #################################
        
        self.initialize_misc(f'{args.supernet_train_method}-{args.seed}')
        if self.resume_from_checkpoint():
            logging.info('Resume the policy weights...')
        else:
            self.resume_from_checkpoint(pretrain_checkpoint)
            logging.info('Resume the pre-trained super-net to the best kdt epochs')
        
        args.save_every_epoch = 10
        # use a policy epochs to indicate after super-net training how long we will enforce the landmark loss!        
        # save the policy training
        # tries out different landmarks? 
        for epoch in range(self.epoch, args.epochs + self.policy_epochs):
            self.epoch = epoch
            scheduler.step()
            lr = scheduler.get_lr()[0]
            epoch_start = f'epoch {epoch} lr {lr}' 
            
            # DARTS train
            if args.supernet_train_method == 'darts_rankloss':
                logging.info(epoch_start + f'  ---> {self.args.search_policy} with landmarks')
            else:
                # we can keep enforce the landmark fn later, but let's do a first baseline for now.
                logging.info(epoch_start + f'  ---> {self.args.search_policy} withOUT landmarks')
            
            train_acc, train_obj = self.train_fn(
                train_queue, valid_queue, parallel_model, train_criterion, optimizer, lr, 
                architect=self.architect
            )

            logging.info(f'Arch Parameter Softmax: \n'
                            f'{nn.functional.softmax(self.model.arch_parameters[0], dim=-1)}')
            self.logging_fn(train_acc, train_obj, epoch, 'Train', display_dict={'lr': lr})

            valid_acc, valid_obj = self.validate_model(
                parallel_model, valid_queue, self.model_spec_id, self.model_spec
            )
            self.logging_fn(valid_acc, valid_obj, epoch, 'Valid')

            self.save_checkpoint(parallel_model, optimizer)
            # utils.save_checkpoint(parallel_model, optimizer, self.running_stats, self.exp_dir)

            # end of normal training loop.
            if not self.check_should_save(epoch):
                continue
            # Saving the current performance of arch pool into the list.
            # Evaluate seed archs
            
            logging.info("="*89)
            logging.info('Evalutation at epoch {}'.format(epoch))
            
            # update the eval list
            # only sample the architecture within the search space to evaluate, otherwise nonsense.
            # as kdt cannot be computed at all.
            ids, sample_nums = self.sample_and_summarize(10, within_search_space=True)
            for i in ids:
                if i: # push only in the search space...
                    self.search_space.eval_model_spec_id_append(i)
            self.evaluate(epoch, test_queue, fitnesses_dict=None, train_queue=train_queue)

            self.save_duplicate_arch_pool('valid', epoch)
            self.save_results(epoch, rank_details=True)
            if len(ids) > 0:
                self.save_policy_results(epoch, ids, sample_nums)

        logging.info(">> Search phase finished! Evaluate here for better results.")
        # add later, return the model specs that is evaluated across the time.
        # Process the ranking in the end, re
        # turn the best of training.
        return self.post_search_phase()

    def post_search_phase(self):
        # do something useful here
        # samples the architecture alphas.
        # spec = self.model.genotype()
        # _id = self.search_space.topology_to_id(spec)
        ids, sample_nums = self.sample_and_summarize(10, within_search_space=False)

        # make this to ids
        arch_pool = [self.search_space.topologies[i] if isinstance(i, int) else i for i in ids]

        # ids is model_spec correspondingly.
        arch_result = self.evaluate(self.epoch, self._datasets[2], None, None, model_spec_id_pool=arch_pool)
        self.save_results(self.epoch, rank_details=False)
        self.save_policy_results(self.epoch, ids, sample_nums)
        # key of eval_results is either a model_id or a model_spec.hash_spec()
        # rank the search policy and save to post-SEARCH

        arch_accs = []
        arch_ids = []
        for ind, spec in enumerate(arch_pool):
            if self.args.debug and ind > 10:
                break
            _id = self.search_space.topology_to_id(spec)
            if _id is None:
                _id = spec.hash_spec()
            arch_accs.append(arch_result[_id][0])
            arch_ids.append(_id)
            # logging.info(f'Architecture {_id} with evaluation result {arch_result[_id]}')

        final_accs, final_archs, final_ids = zip(*sorted(zip(arch_accs, arch_pool, arch_ids))[::-1])
        self.save_arch_pool_performance(final_archs, final_accs, prefix='POST-SEARCH')
        s = ', '.join(map(str, zip(final_accs[:5], final_archs[:5])))
        logging.info(f'Final architecuture with performances \n {s}')
        return [i if isinstance(i, int) else -1 for i in final_ids[:5]], final_archs[:5]

    def sample_and_summarize(self, top_k=10, within_search_space=True):
        """Sample the differentiable sampler.

        Parameters
        ----------
        top_k : int, optional
            [description], by default 10
        within_search_space : bool, optional
            If True, always return the architectures within the search space
            Else, return the model_spec
            by default True

        Returns
        -------
        [type]
            [description]
        """
        if within_search_space:
            arch_count = dict()
            for _ in range(self.top_K_num_sample):
                genotype = self.model.genotype(method='random')
                ind = self.search_space.topology_to_id(genotype)
                if ind:
                    if ind in arch_count.keys():
                        arch_count[ind] += 1
                    else:
                        arch_count[ind] = 1
        else:
            arch_count = dict()
            for _ in range(self.top_K_num_sample):
                genotype = self.model.genotype(method='random')
                ind = genotype
                if ind:
                    if ind in arch_count.keys():
                        arch_count[ind] += 1
                    else:
                        arch_count[ind] = 1

        if len(arch_count) == 0:
            return [], []

        # sort the numbers
        l = [(k,v) for k, v in arch_count.items()]
        l.sort(key=lambda x: x[1])
        l = l[::-1]
        ids, sample_nums = zip(*l)
        ids = ids[:self.top_K_complete_evaluate]
        sample_nums = sample_nums[:self.top_K_complete_evaluate]
        logging.info('Top {} Confidence: \n{} \n{}'.format(top_k, ids[:top_k], sample_nums[:top_k]))
        return ids[:top_k], sample_nums[:top_k]
    
    def resume_from_checkpoint(self, path=None, epoch=None):
        res_dict = super().resume_from_checkpoint(path, epoch)
        # add support for save checkpoint
        # if os.path.exists(self.exp_dir + '/architect.pt'):
        #     self.architect.optimizer.load_state_dict(torch.load(self.exp_dir + '/architect.pt'))
        if res_dict and 'architect' in res_dict.keys():
            self.architect.optimizer.load_state_dict(res_dict['architect'])
            logging.info(f'Resuming {self.args.search_policy} Architect.')
        return res_dict

    def save_checkpoint(self, model, optimizer, backup_epoch=None, other_dict=None):
        other_dict = other_dict or dict()
        other_dict['architect'] = self.architect.optimizer.state_dict()
        return super().save_checkpoint(model, optimizer, backup_epoch, other_dict)
        