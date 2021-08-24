import logging
import operator
import os
from collections import namedtuple
import numpy as np
import gc
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.utils.tensorboard import SummaryWriter

import utils as project_utils
import nasws.cnn.utils
from nasws.cnn.search_space.nasbench101.util import change_model_spec
from nasws.cnn.search_space.nasbench101.model import NasBenchNet
import nasws.cnn.policy.nao_policy.utils as nao_utils
from  nasws.cnn.policy.nao_policy.utils_for_nasbench import NASBench101NAOParsing, NASBench101NAOParsing_v1
from visualization.process_data import tensorboard_summarize_list
from nasws.cnn.procedures.train_search_procedure import nao_model_validation_nasbench
from ..cnn_general_search_policies import CNNSearchPolicy

from .model import NASNetworkCIFAR
from .model_search import NASWSNetworkCIFAR
from .model_search_nasbench import NasBenchNetSearchNAO

from .controller import NAO_Darts, NAO_Nasbench
Rank = namedtuple('Rank', 'valid_acc valid_obj geno_id gt_rank')


class NAONasBenchSearch(CNNSearchPolicy):

    top_K_complete_evaluate = 100

    # nao_search_config
    def wrap_nao_search_config(self):
        for k, v in self.args.nao_search_config.__dict__.items():
            self.args.__dict__[k] = v

    def initialize_model(self):
        """
        Initialize model, may change across different model.
        :return:
        """
        parallel_model, optimizer, scheduler = super().initialize_model()
        args = self.args
        if self.args.search_space == 'nasbench':
            utils = NASBench101NAOParsing_v1()
            # utils = nao_nasbench_utils
            nao = NAO_Nasbench(
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
                args=args,
            )
        else:
            utils = nao_utils
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
            )

        nao = nao.cuda()
        logging.info("Encoder-Predictor-Decoder param size = %fMB", utils.count_parameters_in_MB(nao))

        self.controller = nao
        return model, optimizer, scheduler, nao

    def run(self):
        args = self.args
        self.utils = nao_nasbench_utils if args.search_space == 'nasbench' else nao_utils
        utils = self.utils
        self.nao_search_config = args.nao_search_config
        self.wrap_nao_search_config()
        train_queue, valid_queue, test_queue, train_criterion, eval_criterion = self.initialize_run()

        args.steps = int(np.ceil(45000 / args.child_batch_size)) * args.child_epochs
        if args.child_arch_pool is not None:
            logging.info('Architecture pool is provided, loading')
            with open(args.child_arch_pool) as f:
                archs = f.read().splitlines()
                archs = list(map(self.utils.build_dag, archs))
                child_arch_pool = archs
        elif os.path.exists(os.path.join(self.exp_dir, 'arch_pool')):
            logging.info('Architecture pool is founded, loading')
            with open(os.path.join(self.exp_dir, 'arch_pool')) as f:
                archs = f.read().splitlines()
                archs = list(map(self.utils.build_dag, archs))
                child_arch_pool = archs
        else:
            child_arch_pool = None

        child_eval_epochs = eval(args.child_eval_epochs)

        # build the functions.
        args = self.args

        model, optimizer, scheduler, nao = self.initialize_model()
        fitness_dict = {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Train child model
        if child_arch_pool is None:
            logging.info('Architecture pool is not provided, randomly generating now')
            child_arch_pool = self.generate_new_arch(args.controller_seed_arch)
        child_arch_pool_prob = self.child_arch_pool_prob(child_arch_pool)
        eval_points = self.utils.generate_eval_points(child_eval_epochs, 0, args.child_epochs)
        logging.info("eval epochs = %s", eval_points)

        step = 0
        # Begin the full loop
        for epoch in range(args.epochs):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            # sample an arch to train
            train_acc, train_obj, step = self.child_train(train_queue, model, optimizer, step, child_arch_pool,
                                                            child_arch_pool_prob, train_criterion)
            if self.writer:
                self.writer.add_scalar(f'train/loss', train_obj, epoch)
                self.writer.add_scalar(f'train/top_1_acc', train_acc, epoch)

            logging.info('train_acc %f', train_acc)

            if epoch not in eval_points:
                # Continue training the architectures
                continue

            # Evaluate seed archs
            valid_accuracy_list = self.child_valid(valid_queue, model, child_arch_pool, eval_criterion)

            # Output archs and evaluated error rate
            old_archs = child_arch_pool
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

            if epoch == args.child_epochs:
                break

            # Train Encoder-Predictor-Decoder
            logging.info('Training Encoder-Predictor-Decoder')
            min_val = min(old_archs_perf)
            max_val = max(old_archs_perf)
            encoder_target = [(i - min_val) / (max_val - min_val) for i in old_archs_perf]
            encoder_input = self.process_arch_to_seq(old_archs)
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
            nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True,
                                                 swap=True if args.controller_expand is None else False)
            nao_valid_dataset = utils.NAODataset(valid_encoder_input, valid_encoder_target, False)
            nao_train_queue = torch.utils.data.DataLoader(
                nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
            nao_valid_queue = torch.utils.data.DataLoader(
                nao_valid_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True)
            nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.controller_lr,
                                             weight_decay=args.controller_l2_reg)

            # train the sampler.
            for nao_epoch in range(1, args.controller_epochs + 1):
                nao_loss, nao_mse, nao_ce = self.nao_train(nao_train_queue, nao, nao_optimizer)
                logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
                if nao_epoch % 100 == 0:
                    pa, hs = self.nao_valid(nao_valid_queue, nao)
                    logging.info("Evaluation on valid data")
                    logging.info('epoch %04d pairwise accuracy %.6f hamming distance %.6f', epoch, pa, hs)
                    self.writer.add_scalar(f'valid/hamming_distance', hs, epoch)
                    self.writer.add_scalar(f'valid/pairwise_acc', pa, epoch)

            # Generate new archs
            new_archs = []
            max_step_size = 50
            predict_step_size = 0
            top100_archs = self.process_arch_to_seq(old_archs[:100])
            nao_infer_dataset = utils.NAODataset(top100_archs, None, False)
            nao_infer_queue = torch.utils.data.DataLoader(
                nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
            while len(new_archs) < args.controller_new_arch:
                predict_step_size += 1
                logging.info('Generate new architectures with step size %d', predict_step_size)
                new_arch = self.nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
                for arch in new_arch:
                    if arch not in encoder_input and arch not in new_archs:
                        new_archs.append(arch)
                    if len(new_archs) >= args.controller_new_arch:
                        break
                logging.info('%d new archs generated now', len(new_archs))
                if predict_step_size > max_step_size:
                    break
            # new_archs = list(map(lambda x: utils.parse_seq_to_arch(x, 2), new_archs))
            new_archs = self.process_seq_to_arch(new_archs)
            num_new_archs = len(new_archs)
            logging.info("Generate %d new archs", num_new_archs)
            # replace bottom archs
            if args.controller_replace:
                new_arch_pool = old_archs[:len(old_archs) - (num_new_archs + args.controller_random_arch)] + \
                                new_archs + self.generate_new_arch(args.controller_random_arch)
            # discard all archs except top k
            elif args.controller_discard:
                new_arch_pool = old_archs[:100] + new_archs + self.generate_new_arch(args.controller_random_arch)
            # use all
            else:
                new_arch_pool = old_archs + new_archs + self.generate_new_arch(args.controller_random_arch)
            logging.info("Totally %d architectures now to train", len(new_arch_pool))

            child_arch_pool = new_arch_pool
            with open(os.path.join(self.exp_dir, 'arch_pool'), 'w') as f:
                for arch in new_arch_pool:
                    f.write('{}\n'.format(self.process_archname(arch)))

            child_arch_pool_prob = self.child_arch_pool_prob(child_arch_pool)

            if epoch % self.args.save_every_epoch == 0:
                project_utils.save_checkpoint(model, optimizer, epoch, self.exp_dir)

        # add later, return the model specs that is evaluated across the time.
        # Process the ranking in the end, return the best of training.
        # IPython.embed(header="Pause for nothing.")
        fitness_dict = self.evaluate(epoch, test_queue,
                                     fitnesses_dict=fitness_dict,
                                     arch_pool=child_arch_pool,
                                     train_queue=train_queue,
                                     criterion=eval_criterion)
        project_utils.save_checkpoint(model, optimizer, epoch, self.exp_dir)
        self.save_results(epoch, rank_details=True)
        ep_k = [k for k in self.ranking_per_epoch.keys()][-1]
        best_id = self.ranking_per_epoch[ep_k][-1][1].geno_id
        return best_id, self.nasbench_model_specs[best_id]

    def child_train(self, train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion):
        utils = self.utils
        objs = nasws.cnn.utils.AverageMeter()
        top1 = nasws.cnn.utils.AverageMeter()
        top5 = nasws.cnn.utils.AverageMeter()
        model.train()
        for step, (input, target) in enumerate(train_queue):
            if self.args.debug:
                if step > 10:
                    print("Break after 10 batch")
                    break
            input = input.cuda().requires_grad_()
            target = target.cuda()

            optimizer.zero_grad()

            # sample an arch to train
            arch = utils.sample_arch(arch_pool, arch_pool_prob)
            arch_l = arch
            arch = self.process_arch(arch)
            logits, aux_logits = model(input, arch, global_step, bn_train=False)
            global_step += 1
            loss = criterion(logits, target)
            if aux_logits is not None:
                aux_loss = criterion(aux_logits, target)
                loss += 0.4 * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.args.child_grad_bound)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if (step + 1) % 100 == 0:
                logging.info('Train %03d loss %e top1 %f top5 %f', step + 1, objs.avg, top1.avg, top5.avg)
                logging.info('Arch: %s', self.process_archname(arch_l))

        return top1.avg, objs.avg, global_step

    def child_valid(self,valid_queue, model, arch_pool, criterion):
        valid_acc_list = []
        with torch.no_grad():
            model.eval()
            for i, arch in enumerate(arch_pool):
                # for step, (inputs, targets) in enumerate(valid_queue):
                inputs, targets = next(iter(valid_queue))
                inputs = inputs.cuda()
                targets = targets.cuda()
                arch_l = arch
                arch = self.process_arch(arch)
                logits, _ = model(inputs, arch, bn_train=True)
                loss = criterion(logits, targets)

                prec1, prec5 = self.utils.accuracy(logits, targets, topk=(1, 5))
                valid_acc_list.append(prec1.data / 100)

                if (i + 1) % 100 == 0:
                    logging.info('Valid arch %s\n loss %.2f top1 %f top5 %f', self.process_archname(arch_l),
                                 loss, prec1, prec5)

        return valid_acc_list

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
        new_arch_list = []
        model.eval()
        for i, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_input = encoder_input.cuda()
            model.zero_grad()
            new_arch = model.generate_new_arch(encoder_input, step, direction=direction)
            new_arch_list.extend(new_arch.data.squeeze().tolist())
        return new_arch_list

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
        if args.search_space == 'nasbench':
            for _ in range(args.controller_expand - 1):
                for src, tgt in zip(encoder_input[:split], encoder_target[:split]):
                    train_encoder_input.append(src)
                    train_encoder_target.append(tgt)

        else:
            for _ in range(args.controller_expand - 1):
                # TODO what is controller expand?????
                for src, tgt in zip(encoder_input[:split], encoder_target[:split]):
                    a = np.random.randint(0, args.child_nodes)
                    b = np.random.randint(0, args.child_nodes)
                    src = src[:4 * a] + src[4 * a + 2:4 * a + 4] + \
                          src[4 * a:4 * a + 2] + src[4 * (a + 1):20 + 4 * b] + \
                          src[20 + 4 * b + 2:20 + 4 * b + 4] + src[20 + 4 * b:20 + 4 * b + 2] + \
                          src[20 + 4 * (b + 1):]
                    train_encoder_input.append(src)
                    train_encoder_target.append(tgt)

        return train_encoder_input, train_encoder_target, valid_encoder_input, valid_encoder_target

    # def process_arch(self,arch):
    #     if self.args.search_space == 'nasbench':
    #         matrix, ops = nao_nasbench_utils.parse_arch_to_model_spec_matrix_op(arch, self.args.child_nodes)
    #         model_spec = ModelSpec_v2(matrix, ops)
    #         return model_spec
    #     else:
    #         return arch

    def process_arch_to_seq(self, old_archs):
        if self.args.search_space =='nasbench':
            encoder_input =list(map(lambda x: self.utils.parse_arch_to_seq(x, 2, self.args.child_nodes), old_archs))
        else:
            encoder_input = list(
                map(lambda x: self.utils.parse_arch_to_seq(x[0], 2) + self.utils.parse_arch_to_seq(x[1], 2), old_archs))

        return encoder_input

    def process_seq_to_arch(self, old_archs):
        if self.args.search_space =='nasbench':
            encoder_input =list(map(lambda x: self.utils.parse_seq_to_arch(x, 2, self.args.child_nodes), old_archs))
        else:
            encoder_input = list(
                map(lambda x: self.utils.parse_seq_to_arch(x[0], 2) + self.utils.parse_seq_to_arch(x[1], 2), old_archs))

        return encoder_input

    def process_archname(self, arch):
        if self.args.search_space == 'nasbench':
            return ' '.join(map(str, arch))
        else:
            return ' '.join(map(str, arch[0] + arch[1]))

    def generate_new_arch(self, num_new):
        num_ops = 3 if self.args.search_space == 'nasbench' else 5
        child_arch_pool = self.utils.generate_arch(num_new, self.args.child_nodes, num_ops)
        return child_arch_pool

    def evaluate(self, epoch, data_source, arch_pool=None, fitnesses_dict=None, train_queue=None, criterion=None):
        """
        Full evaluation of all possible models.
        :param epoch:
        :param data_source:
        :param fitnesses_dict: Store the model_spec_id -> accuracy
        :return:
        """

        fitnesses_dict = fitnesses_dict or {}
        total_avg_acc = 0
        total_avg_obj = 0

        # rank dict for the possible solutions
        model_specs_rank = {}
        model_specs_rank_before = {}
        queries = {}
        # as backup
        ind = 0
        eval_result = {}
        # let us sample 200 architecture to evaluate. # just keep the top K.
        clean_arch_pool = self.clean_arch_pool(arch_pool)[:self.top_K_complete_evaluate]

        while ind < len(clean_arch_pool):
            # get this id
            if self.args.debug and ind > 10:
                break

            arch = clean_arch_pool[ind]
            new_model_spec = self.utils.parse_arch_to_model_spec(clean_arch_pool[ind])
            ind += 1  # increment this.
            try:
                model_spec_id = self.nasbench_hashs.index(new_model_spec.hash_spec())
            except Exception as e:
                logging.error(e)
                continue

            query = {'test accuracy':self.search_space.nasbench.perf_rank[model_spec_id]}

            # selecting the current subDAG in our DAG to train
            change_model_spec(self.parallel_model, new_model_spec)
            # Reset the weights.
            # evaluate before train
            self.logger.info('evaluate the model spec id: {}'.format(model_spec_id))
            _avg_val_acc, _avg_val_acc5, _avg_val_obj = self.child_test(data_source, self.parallel_model, arch, criterion=criterion)
            eval_result[model_spec_id] = _avg_val_acc, _avg_val_obj

            logging.info("Query: {}".format(query))
            # update the total loss.
            total_avg_acc += _avg_val_acc
            total_avg_obj += _avg_val_obj

            # saving the particle fit in our dictionaries
            fitnesses_dict[model_spec_id] = _avg_val_acc
            ms_hash = self.nasbench_hashs[model_spec_id]
            model_specs_rank[ms_hash] = Rank(_avg_val_acc, _avg_val_obj, model_spec_id,
                                              self.search_space.rank_by_mid[model_spec_id])
            queries[ms_hash] = query
            gc.collect()

        # save the ranking, according to their GENOTYPE but not particle id
        rank_gens = sorted(model_specs_rank.items(), key=operator.itemgetter(1))

        self.ranking_per_epoch[epoch] = rank_gens
        self.eval_result[epoch] = eval_result
        # IPython.embed(header="Check evaluation result")

        self.logger.info('VALIDATION RANKING OF PARTICLES')
        for pos, elem in enumerate(rank_gens):
            self.logger.info(f'particle gen id: {elem[1].geno_id}, acc: {elem[1].valid_acc}, obj {elem[1].valid_obj}, '
                             f'hash: {elem[0]}, pos {pos}')

        if self.writer:
            # process data into list.
            accs_after, objs_after = zip(*eval_result.values())
            tensorboard_summarize_list(accs_after, writer=self.writer, key='neweval_after/acc', step=epoch, ascending=False)
            tensorboard_summarize_list(objs_after, writer=self.writer, key='neweval_after/obj', step=epoch)

        return fitnesses_dict

    def process_nasbench(self):
        super(NAONasBenchSearch, self).process_nasbench(only_hash=False)

    def child_test(self, test_queue, model, arch, criterion, verbose=True):
        utils = self.utils
        objs = nasws.cnn.utils.AverageMeter()
        top1 = nasws.cnn.utils.AverageMeter()
        top5 = nasws.cnn.utils.AverageMeter()
        model.eval()
        # arch_l = arch
        arch = self.utils.parse_arch_to_model_spec(arch)
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
                    logging.info('test | step %03d | loss %e | acc %f | acc-5 %f', step, objs.avg, top1.avg,
                                 top5.avg)

        return top1.avg, top5.avg, objs.avg

    def clean_arch_pool(self, arch_pool):
        new_arch_pool = []
        for i in arch_pool:
            if not i in new_arch_pool:
                new_arch_pool.append(i)

        return new_arch_pool

    def child_arch_pool_prob(self, child_arch_pool):
        # raise NotImplementedError('do not support params for now ...')
        args = self.args
        if args.child_sample_policy == 'params':
            child_arch_pool_prob = []
            print('num is %d', len(child_arch_pool))
            for arch in child_arch_pool:
                # just to count the parameters and use as prob, kind of a biased sampling.
                # use model hash to query.
                if self.args.search_space == 'nasbench':
                    tmp_model = self.fixmodel_fn(3, self.process_arch(arch), args.nasbench_config)
                else:
                    tmp_model = self.fixmodel_fn(args, 10, args.child_layers, args.child_nodes, args.child_channels,
                                                args.child_keep_prob, args.child_drop_path_keep_prob,
                                                args.child_use_aux_head, args.steps, arch)
                    
                child_arch_pool_prob.append(self.utils.count_parameters_in_MB(tmp_model))
                del tmp_model
        else:
            child_arch_pool_prob = None
        return child_arch_pool_prob

    