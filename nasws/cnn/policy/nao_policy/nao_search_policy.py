import copy
import logging
import operator
import os
from collections import namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset

import nasws.cnn.utils
import utils as project_utils
from nasws.cnn.search_space.nasbench101.util import change_model_spec
from nasws.cnn.search_space.nasbench101.model import NasBenchNet
import nasws.cnn.policy.nao_policy.utils as nao_utils

from .configs import naoargs, default_nao_search_configs
from .controller import NAO_Darts, NAO_Nasbench, NAO_Nasbench201
from . import utils_for_nasbench, utils_for_darts, utils_for_nasbench201
from .model import NASNetworkCIFAR
from .model_search import NASWSNetworkCIFAR


from nasws.cnn.search_space.nasbench101.nasbench_api_v2 import ModelSpec_v2
from visualization.process_data import tensorboard_summarize_list
import gc

Rank = namedtuple('Rank', 'valid_acc valid_obj geno_id gt_rank')
# from nasws.cnn.policy.random_policy import NasBenchWeightSharingPolicy
from .model_search_nasbench import NasBenchNetSearchNAO
from nasws.cnn.procedures.train_search_procedure import nao_model_validation_nasbench
from nasws.cnn.policy.cnn_general_search_policies import CNNSearchPolicy




class NAOSearchPolicy(CNNSearchPolicy):
    """ NAO support search policy

    Some notations:
        arch: string based architecture encoding, used for human understanding...
        seq:  mapping from arch to a sequence, the actual input to NAO controller

    """
    top_K_complete_evaluate = 100
    _child_arch_pool = None
    _child_arch_pool_prob = None
    nao_search_config = project_utils.DictAttr(naoargs.__dict__)
    
    # nao_search_config
    def wrap_nao_search_config(self):
        for k, v in self.nao_search_config.__dict__.items():
            self.args.__dict__[k] = v

    def initialize_model(self):
        """
        Initialize model, may change across different model.
        :return:
        """
        model, optimizer, scheduler = super().initialize_model(False)
        args = self.args
        nao_search_config = self.nao_search_config 
        args.nao_search_config = default_nao_search_configs(nao_search_config, args)
        self.wrap_nao_search_config()

        if 'nasbench101' in self.args.search_space:

            nao = NAO_Nasbench(
                args.controller_encoder_layers,
                args.controller_mlp_layers,
                args.controller_decoder_layers,
                args.controller_encoder_vocab_size,
                args.controller_encoder_hidden_size,
                args.controller_mlp_hidden_size,
                args.controller_mlp_dropout,
                args.controller_encoder_length,
                args.controller_source_length,
                args.controller_decoder_length,
                args
            )
        elif 'nasbench201' in self.args.search_space:
            nao = NAO_Nasbench201(
                args.controller_encoder_layers,
                args.controller_mlp_layers,
                args.controller_decoder_layers,
                args.controller_encoder_vocab_size,
                args.controller_encoder_hidden_size,
                args.controller_mlp_hidden_size,
                args.controller_mlp_dropout,
                args.controller_encoder_length,
                args.controller_source_length,
                args.controller_decoder_length,
                args
            )
        else:
            nao = NAO_Darts(
                args.controller_encoder_layers,
                args.controller_encoder_vocab_size,
                args.controller_encoder_hidden_size,
                args.controller_encoder_dropout,
                args.controller_encoder_length,
                args.controller_source_length,
                args.controller_encoder_emb_size,
                args.controller_mlp_layers,
                args.controller_mlp_hidden_size,
                args.controller_mlp_dropout,
                args.controller_decoder_layers,
                args.controller_decoder_vocab_size,
                args.controller_decoder_hidden_size,
                args.controller_decoder_dropout,
                args.controller_decoder_length,
                args
            )

        nao = nao.cuda()
        logging.info("Encoder-Predictor-Decoder param size = {}MB".format(project_utils.count_parameters_in_MB(nao)))

        self.nao = nao
        if args.resume:
            self.resume_from_checkpoint()
        return model, optimizer, scheduler, nao
        
    @property
    def child_arch_pool_prob(self, pool=None):
        if self._child_arch_pool_prob is None:
            self._child_arch_pool_prob = dict()

        child_arch_pool = pool or self.child_arch_pool
        args = self.args
        child_arch_pool_prob = []

        for arch in child_arch_pool:
            h = self.process_archname(arch)
            if h in self._child_arch_pool_prob:
                prob = self._child_arch_pool_prob[h]
            else:
                # no prob, compute a new one
                if args.child_sample_policy == 'params':
                    # just to count the parameters and use as prob, kind of a biased sampling.
                    # use model hash to query.
                    if 'nasbench101' in self.args.search_space:
                        tmp_model = self.search_space.topology_fn(3, self.arch_to_model_spec(arch), args.nasbench_config, args=copy.deepcopy(args))
                    elif 'nasbench201' in self.args.search_space:
                        tmp_model = self.search_space.topology_fn(self.num_classes, self.utils.parse_arch_to_model_spec(arch))
                    else:
                        tmp_model = self.search_space.topology_fn(
                            args.child_channels, self.num_classes, args.child_layers, False, self.utils.parse_arch_to_model_spec(arch)._darts_genotype,
                            args=args)
                    prob = nao_utils.count_parameters_in_MB(tmp_model)
                    del tmp_model
                else:
                    prob = 1
                self._child_arch_pool_prob[h] = prob
            child_arch_pool_prob.append(prob)

        return child_arch_pool_prob

    @property
    def child_arch_pool(self):
        # init arch pool
        if self._child_arch_pool is None:
            if self.args.child_arch_pool is not None:
                logging.info('Architecture pool is provided, loading')
                with open(self.args.child_arch_pool) as f:
                    archs = f.read().splitlines()
                    archs = list(map(self.utils.deserialize_arch, archs))
                    child_arch_pool = archs
            elif os.path.exists(os.path.join(self.exp_dir, 'arch_pool')):
                logging.info('Architecture pool is founded, loading')
                with open(os.path.join(self.exp_dir, 'arch_pool')) as f:
                    archs = f.read().splitlines()
                    archs = list(map(self.utils.deserialize_arch, archs))
                    child_arch_pool = archs
            else:
                child_arch_pool = []
            self._child_arch_pool = child_arch_pool
        
        return self._child_arch_pool

    @child_arch_pool.setter
    def child_arch_pool(self, pool):
        # avoid the duplicated architectures...
        self._child_arch_pool = []
        # keep the order...
        for a in pool:
            if a not in self._child_arch_pool:
                self._child_arch_pool.append(a)

    def run(self):
        args = self.args
        self.wrap_nao_search_config()

        train_queue, valid_queue, test_queue, train_criterion = self.initialize_run()
        eval_criterion = self.eval_criterion
        args.steps = int(np.ceil(len(train_queue) / args.child_batch_size)) * args.child_epochs
        # build the functions.
        args = self.args
        model, optimizer, scheduler, nao = self.initialize_model()
        fitness_dict = {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Train child model
        if len(self.child_arch_pool) == 0:
            logging.info('Architecture pool is not provided, randomly generating now')
            self.child_arch_pool = self.generate_new_arch(args.controller_seed_arch)

        eval_points = nao_utils.generate_eval_points(
            args.supernet_warmup_epoch, args.child_stand_alone_epoch, args.epochs)
        logging.info("Eval / NAO train epochs = {}".format(eval_points))
        # import ipdb; ipdb.set_trace()
        # Begin the full loop
        for epoch in range(self.epoch, args.epochs):
            
            lr = scheduler.get_last_lr()[0]
            self.epoch = epoch # update epoch for other policy settings
            
            train_acc, train_obj = self.train_fn(train_queue, valid_queue, model, train_criterion, optimizer, lr)
            self.logging_fn(train_acc, train_obj, epoch, 'Train', display_dict={'lr': lr})
            scheduler.step()
            # validation
            _,valid_acc, valid_obj = self.child_valid(valid_queue, model, self.child_arch_pool, eval_criterion)
            self.logging_fn(valid_acc, valid_obj, epoch, 'Valid')

            self.save_checkpoint(model, optimizer)
            

            if self.check_should_save(epoch):
                # normal validation here!
                if 'imagenet' in self.args.dataset:
                    # duplicate for imagenet for easy recovery...
                    self.save_checkpoint(model, optimizer, epoch)
                model_spec_pool = [
                    self.utils.parse_arch_to_model_spec(arch) for arch in self.child_arch_pool
                ]
                fitness_dict = self.evaluate(epoch, test_queue, fitnesses_dict=fitness_dict, train_queue=train_queue, model_spec_id_pool=model_spec_pool)
                self.save_results(epoch, rank_details=True)

            if epoch not in eval_points:
                # Continue training the architectures
                continue

            # Evaluate seed archs
            valid_accuracy_list,_,_ = self.child_valid(valid_queue, model, self.child_arch_pool, eval_criterion)

            # Output archs and evaluated error rate
            old_archs = self.child_arch_pool
            old_archs_perf = valid_accuracy_list

            old_archs_sorted_indices = np.argsort(old_archs_perf)[::-1]
            old_archs = [old_archs[i] for i in old_archs_sorted_indices]
            old_archs_perf = [old_archs_perf[i] for i in old_archs_sorted_indices]

            with open(os.path.join(self.exp_dir, 'arch_pool.{}'.format(epoch)), 'w') as fa:
                with open(os.path.join(self.exp_dir, 'arch_pool.perf.{}'.format(epoch)), 'w') as fp:
                    with open(os.path.join(self.exp_dir, 'arch_pool'), 'w') as fa_latest:
                        with open(os.path.join(self.exp_dir, 'arch_pool.perf'), 'w') as fp_latest:
                            for arch, perf in zip(old_archs, old_archs_perf):
                                arch = self.process_archname(arch)
                                fa.write('{}\n'.format(arch))
                                fa_latest.write('{}\n'.format(arch))
                                fp.write('{}\n'.format(perf))
                                fp_latest.write('{}\n'.format(perf))

            # Train Encoder-Predictor-Decoder
            logging.info('==== Training Encoder-Predictor-Decoder ===')
            # normalize the performance predictor by the min and max of each batch sampler...
            min_val = min(old_archs_perf)
            max_val = max(old_archs_perf)
            encoder_target = [(i - min_val) / (max_val - min_val) for i in old_archs_perf]
            encoder_input = self.process_archs_to_seqs(old_archs)

            if args.controller_expand is not None:
                train_encoder_input, train_encoder_target, valid_encoder_input, valid_encoder_target = \
                    self.expand_controller(encoder_input, encoder_target)
            else:
                train_encoder_input = encoder_input
                train_encoder_target = encoder_target
                valid_encoder_input = encoder_input
                valid_encoder_target = encoder_target

            logging.info('Train data: {}\tValid data: {}'.format(len(train_encoder_input), len(valid_encoder_input)))

            # gerenrate NAO dataset.
            nao_train_dataset = nao_utils.NAODataset(train_encoder_input, train_encoder_target, True)
            nao_valid_dataset = nao_utils.NAODataset(valid_encoder_input, valid_encoder_target, False)
            nao_train_queue = torch.utils.data.DataLoader(
                nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
            nao_valid_queue = torch.utils.data.DataLoader(
                nao_valid_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True)
            nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.controller_lr,
                                             weight_decay=args.controller_l2_reg)

            # train the sampler.
            for nao_epoch in range(1, args.controller_epochs + 1):
                nao_loss, nao_mse, nao_ce = self.nao_train(nao_train_queue, nao, nao_optimizer)
                logging.info(f"epoch {nao_epoch:04d} train loss {nao_loss:.6f} mse {nao_mse:.6f} ce {nao_ce:.6f}")
                if nao_epoch % 100 == 0:
                    pa, hs = self.nao_valid(nao_valid_queue, nao)
                    logging.info("Evaluation on valid data")
                    logging.info(f'epoch {epoch} pairwise accuracy {pa:.6f} hamming distance {hs:.6f}')
                    self.writer.add_scalar(f'valid/hamming_distance', hs, epoch)
                    self.writer.add_scalar(f'valid/pairwise_acc', pa, epoch)

            new_arch_pool, _ = self.nao_generate_new_archs(old_archs)

            # assign this to the child arch pool.
            self.child_arch_pool = new_arch_pool
            with open(os.path.join(self.exp_dir, 'arch_pool'), 'w') as f:
                for arch in new_arch_pool:
                    f.write('{}\n'.format(self.utils.serialize_arch(arch)))
            
        return self.post_search_phase()

    def post_search_phase(self):
        """Adapted for NAO policy. because we do not need to run the random or evolutionary search at all.

        Returns
        -------
        Top 5 architectures reported given the super-net performance.
            IDS, archs
        """
        logging.info('==== Post search phase ===')
        # given the curren pool, we should return the 
        new_arch_pool, _ = self.nao_generate_new_archs(self.child_arch_pool)
        
        model_spec_pool = [
                self.utils.parse_arch_to_model_spec(arch) for arch in new_arch_pool
        ]
        _,_, test_queue = self.load_dataset()
        self.evaluate(self.args.epochs, test_queue, model_spec_id_pool=model_spec_pool)
        self.save_results(self.args.epochs)

        arch_result = self.eval_result[self.args.epochs]
        arch_accs = []
        arch_ids = []
        # decode accs into 
        for ind, spec in enumerate(model_spec_pool):
            if self.args.debug and ind > 10:
                break
            _id = self.search_space.topology_to_id(spec)
            if _id is None:
                _id = spec.hash_spec()
            arch_accs.append(arch_result[_id][0])
            arch_ids.append(_id)
            # logging.info(f'Architecture {_id} with evaluation result {arch_result[_id]}')

        final_accs, final_archs, final_ids = zip(*sorted(zip(arch_accs, new_arch_pool, arch_ids))[::-1])
        self.save_arch_pool_performance(final_archs, final_accs, prefix='POST-SEARCH')
        s = ', '.join(map(str, zip(final_accs[:5], final_archs[:5])))
        logging.info(f'Final architecuture with performances \n {s}')
        return final_ids[:5], final_archs[:5]
        
    def nao_generate_new_archs(self, old_archs, encoder_input=None, must_in_search_space=True):
        """Infer new architectures based on NAO result.

        Parameters
        ----------
        old_archs : list
            list of old architectures, basically this is used as a starting point of new architectures.
        encoder_input : list, optional
            list of seqs, not sure this is a must, leave now for legacy reasons.
        must_in_search_space : bool, optional
            True - new spec must be included in the search space with queried results
            False - can generate any possible model spec, if not in space, need to train from scratch...

        Returns
        -------
        tuple
            1. A list of new arch pool with sampling algorithms
            2. A list of generated archs
        """
        encoder_input = encoder_input or self.process_archs_to_seqs(old_archs)

        args = self.args
        # Generate new archs
        new_archs = []            
        max_step_size = 50
        predict_step_size = 0
        top100_archs = self.process_archs_to_seqs(old_archs[:100])
        nao_infer_dataset = nao_utils.NAODataset(top100_archs, None, False)
        nao_infer_queue = torch.utils.data.DataLoader(
            nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
        
        logging.info('Generate new architectures with step size {}'.format(max_step_size))
        while len(new_archs) < args.controller_new_arch:
            predict_step_size += 1
            new_seqs = self.nao_infer(nao_infer_queue, self.nao, predict_step_size, direction='+')

            for arch in self.process_seqs_to_archs(new_seqs): 
                if arch not in encoder_input and arch not in new_archs:
                    spec = self.utils.parse_arch_to_model_spec(arch)
                    
                    if must_in_search_space:
                        if self.search_space.check_valid(spec):
                            new_archs.append(arch)
                    else:
                        new_archs.append(arch)
                if len(new_archs) >= args.controller_new_arch:
                    break
            logging.info(f'{len(new_archs)} new archs generated now')
            if predict_step_size > max_step_size:
                break
        
        # order does not matter here...
        _new_archs, new_archs = new_archs, []
        
        for a in _new_archs:
            if a not in old_archs:
                new_archs.append(a)

        logging.info("Generate {} new archs".format(len(new_archs)))
        # replace bottom archs
        if args.controller_replace:
            new_arch_pool = old_archs[:len(old_archs) - (len(new_archs) + args.controller_random_arch)] + \
                            new_archs + self.generate_new_arch(args.controller_random_arch)
        # discard all archs except top k
        elif args.controller_discard:
            new_arch_pool = old_archs[:100] + new_archs + self.generate_new_arch(args.controller_random_arch)
        # use all do not use all here.
        else:
            new_arch_pool = old_archs + new_archs + self.generate_new_arch(args.controller_random_arch)
        logging.info(f"Totally {len(new_arch_pool)} architectures now to train")        
        # return new_archs
        return new_arch_pool, new_archs

    def random_sampler(self, model, architect, args):
        """ random sampling in NAO is sampled from arch pool only """
                    # sample an arch to train
        if self.epoch < args.supernet_warmup_epoch:
            logging.debug('random sampler - randomNAS')
            # training with all possible weights
            _, spec = self.search_space.random_topology_random_nas()
        else:
            logging.debug('random sampler - NAO arch')
            # training with only a few architecture in the pool...
            arch = self.utils.sample_arch(self.child_arch_pool)
            # arch = self.utils.sample_arch(self.child_arch_pool, self.child_arch_pool_prob)
            spec = self.utils.parse_arch_to_model_spec(arch)
        return self.change_model_spec_fn(model, spec)

    def child_valid(self,valid_queue, model, arch_pool, criterion):
        valid_acc_list = []
        _valid_queue = []
        nb_models = len(arch_pool)
        nb_batch_per_model = max(len(valid_queue) // nb_models, 1)
        if self.args.debug:
            nb_batch_per_model = 10
        
        total_valid_acc = 0.
        total_valid_obj = 0.

        valid_accs = dict()
        valid_objs = dict()
        with torch.no_grad():
            for step, val_d in enumerate(valid_queue):
                _valid_queue.append(val_d)
                if step % nb_batch_per_model == 0 and step > 0:
                    arch_id = int(step / nb_batch_per_model)
                    if arch_id > nb_models - 1:
                        break
                    # validate one architecture
                    arch = arch_pool[arch_id]
                    h = self.utils.serialize_arch(arch)
                    model = self.change_model_spec_fn(model, self.arch_to_model_spec(arch))
                    model.eval()
                    _valid_acc, _valid_obj = self.eval_fn(_valid_queue, model, criterion)
                    logging.debug(f"model arch {arch} acc {_valid_acc} loss {_valid_obj}")
                    # update the metrics
                    total_valid_acc += _valid_acc
                    total_valid_obj += _valid_obj
                    # store the results
                    valid_accs[h] = _valid_acc
                    valid_objs[h] = _valid_obj
                    valid_acc_list.append(_valid_acc)
                    _valid_queue = []
        self.save_arch_pool_performance(
            archs=list(valid_accs.keys()), perfs=list(valid_accs.values()), 
            prefix='valid')

        return valid_acc_list, total_valid_acc / nb_models, total_valid_obj/ nb_models

    def nao_train(self, train_queue, model, optimizer):
        args = self.args
        objs = nasws.cnn.utils.AverageMeter()
        mse = nasws.cnn.utils.AverageMeter()
        nll = nasws.cnn.utils.AverageMeter()
        model.train()
        for step, sample in enumerate(train_queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_input = sample['decoder_input']
            decoder_target = sample['decoder_target']

            encoder_input = encoder_input.cuda()
            encoder_target = encoder_target.cuda().requires_grad_()
            decoder_input = decoder_input.cuda()
            decoder_target = decoder_target.cuda()

            optimizer.zero_grad()
            predict_value, log_prob, arch = model(encoder_input, decoder_input)
            loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
            loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
            loss = args.controller_trade_off * loss_1 + (1 - args.controller_trade_off) * loss_2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.controller_grad_bound)
            optimizer.step()

            n = encoder_input.size(0)
            objs.update(loss.data, n)
            mse.update(loss_1.data, n)
            nll.update(loss_2.data, n)
        return objs.avg, mse.avg, nll.avg

    @staticmethod
    def nao_valid(queue, model):
        pa = nasws.cnn.utils.AverageMeter()
        hs = nasws.cnn.utils.AverageMeter()
        with torch.no_grad():
            model.eval()
            for step, sample in enumerate(queue):
                encoder_input = sample['encoder_input']
                encoder_target = sample['encoder_target']
                decoder_target = sample['decoder_target']

                encoder_input = encoder_input.cuda()
                encoder_target = encoder_target.cuda()
                decoder_target = decoder_target.cuda()

                predict_value, logits, arch = model(encoder_input)
                n = encoder_input.size(0)
                pairwise_acc = nao_utils.pairwise_accuracy(encoder_target.data.squeeze().tolist(),
                                                       predict_value.data.squeeze().tolist())
                hamming_dis = nao_utils.hamming_distance(decoder_target.data.squeeze().tolist(),
                                                     arch.data.squeeze().tolist())
                pa.update(pairwise_acc, n)
                hs.update(hamming_dis, n)
        return pa.avg, hs.avg

    @staticmethod
    def nao_infer(queue, model, step, direction='+'):
        new_seq_list = []
        model.eval()
        for i, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_input = encoder_input.cuda()
            model.zero_grad()
            seqs = model.generate_new_arch(encoder_input, step, direction=direction)
            new_seqs = seqs.data.squeeze().tolist()
            new_seq_list.extend(new_seqs)
        return new_seq_list

    def expand_controller(self, encoder_input, encoder_target):
        args = self.args
        dataset = list(zip(encoder_input, encoder_target))
        n = len(dataset)
        split = int(n * args.ratio)
        np.random.shuffle(dataset)
        encoder_input, encoder_target = list(zip(*dataset))
        train_encoder_input = list(encoder_input[:split])
        train_encoder_target = list(encoder_target[:split])
        valid_encoder_input = list(encoder_input[split:])
        valid_encoder_target = list(encoder_target[split:])
        for _ in range(args.controller_expand - 1):
            for src, tgt in zip(encoder_input[:split], encoder_target[:split]):
                # Augmentation is not supported in this, just repeat the same data by multiple tiems..
                nsrc = self.utils.augmentation(src)
                train_encoder_input.append(nsrc)
                train_encoder_target.append(tgt)

        return train_encoder_input, train_encoder_target, valid_encoder_input, valid_encoder_target

    def child_test(self, test_queue, model, arch, criterion, verbose=True):
        utils = self.utils
        objs = nasws.cnn.utils.AverageMeter()
        top1 = nasws.cnn.utils.AverageMeter()
        top5 = nasws.cnn.utils.AverageMeter()
        model.eval()
        # arch_l = arch
        arch = self.arch_to_model_spec(arch)
        with torch.no_grad():
            for step, (input, target) in enumerate(test_queue):
                if self.args.debug:
                    if step > 10:
                        print("Break after 10 batch")
                        break
                input = input.cuda()
                target = target.cuda()
                logits, _ = model(input, arch, bn_train=False)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % self.args.report_freq == 0 and step > 0 and verbose:
                    logging.info('test | step {} | loss {} | acc {} | acc-5 {}'.format(step, objs.avg, top1.avg,
                                 top5.avg))

        return top1.avg, top5.avg, objs.avg
    """
    TODO Move these architeture functions to a generic place.
    """
    def initialize_search_space(self):
        super().initialize_search_space()
        # given the search space, we should implement these search space related operation here!!!
        if 'nasbench101' in self.args.search_space:
            self.utils = utils_for_nasbench.NASBench101NAOParsing(self.search_space, self.args)
        elif 'darts' in self.args.search_space:
            self.utils = utils_for_darts.NAOParsingDarts(self.search_space, self.args)
        elif 'nasbench201' in self.args.search_space:
            self.utils = utils_for_nasbench201.NAOParsingNASBench201(self.search_space, self.args)
        else:
            raise NotImplementedError('Only support three search spaces.')

    def arch_to_model_spec(self, arch):
        return self.utils.parse_arch_to_model_spec(arch)

    def process_archs_to_seqs(self, old_archs):
        return list(map(lambda x: self.utils.parse_arch_to_seq(x, B=self.args.num_intermediate_nodes), old_archs))

    def process_seqs_to_archs(self, seqs):
        return list(map(lambda x: self.utils.parse_seq_to_arch(x, B=self.args.num_intermediate_nodes), seqs))

    def process_archname(self, arch):
        """ deprecated ... should use the self.utils.serialize_arch"""
        return self.utils.serialize_arch(arch)

    def generate_new_arch(self, num_new):
        return self.utils.generate_arch(num_new, self.args.num_intermediate_nodes + 2)

    def clean_arch_pool(self, arch_pool):
        new_arch_pool = []
        for i in arch_pool:
            if not i in new_arch_pool:
                new_arch_pool.append(i)

        return new_arch_pool
    
    def save_checkpoint(self, model, optimizer, backup_epoch=None, other_dict=None):
        other_dict = other_dict or {'nao': self.nao.state_dict()}
        return super().save_checkpoint(model, optimizer, backup_epoch, other_dict)

    def resume_from_checkpoint(self, path=None, epoch=None):
        res_dict = super().resume_from_checkpoint(path, epoch)
        if res_dict and 'nao' in res_dict.keys():
            self.nao.load_state_dict(res_dict['nao'])
        return res_dict

class NAONasBenchGroundTruthPolicy(NAOSearchPolicy):

    def child_train(self, train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion):
        utils = self.utils
        model.train()
        for step, (input, target) in enumerate(train_queue):
            if self.args.debug:
                if step > 10:
                    print("Break after 10 batch")
                    break

            # sample an arch to train
            arch = utils.sample_arch(arch_pool, arch_pool_prob)
            global_step += 1
            arch_l = arch
            try:
                arch = self.arch_to_model_spec(arch_l)
                r = self.search_space.nasbench.hash_rank.index(arch.hash_spec())
            except ValueError as e:
                continue

            global_step += 1

        return .0, .0, global_step

    def child_test(self, test_queue, model, arch, criterion, verbose=True):
        try:
            if not hasattr(arch, 'hash_spec'):
                arch = self.arch_to_model_spec(arch)
            r = self.search_space.nasbench.hash_rank.index(arch.hash_spec())
        except ValueError as e:
            return 0., 0., 0.
        valid_acc = self.search_space.nasbench.perf_rank[r][0]
        return valid_acc, 0., 0.

    def child_valid(self, valid_queue, model, arch_pool, criterion):
        valid_acc_list = []

        for i, arch in enumerate(arch_pool):
            # for step, (inputs, targets) in enumerate(valid_queue):
            arch_l = arch
            arch = self.arch_to_model_spec(arch)
            try:
                r = self.search_space.nasbench.hash_rank.index(arch.hash_spec())
                valid_acc = self.search_space.nasbench.perf_rank[r][0]
                valid_acc_list.append(valid_acc)
            except ValueError as e:
                valid_acc_list.append(0.)

        return valid_acc_list
