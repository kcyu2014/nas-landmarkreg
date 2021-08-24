from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import time

import os
import sys
import math
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import copy
from nasws.dataset import dataloader
from torch.utils.tensorboard import SummaryWriter

# Customized import
from nasws.rnn.general_search_policies import RNNSearchPolicy
from utils import batchify, save_checkpoint, create_exp_dir
from . import model_search
from .utils import generate_arch, parse_arch_to_seq, parse_seq_to_arch, \
    normalize_target, get_genotype, PRIMITIVES
from . import controller


class NaoSearchPolicy(RNNSearchPolicy):
    def __init__(self, args):
        super(NaoSearchPolicy, self).__init__(args=args)
        self.initialize_search()

    def initialize_search(self):
        if self.args.child_nhidlast < 0:
            self.args.child_nhidlast = self.args.child_emb_size
        if self.args.child_small_batch_size < 0:
            self.args.child_small_batch_size = self.args.child_batch_size

        self.search_dir = 'search_{}_NAO_SEED_{}_{}'.format(self.args.save,
                                                            self.args.seed,
                                                            time.strftime("%Y%m%d-%H%M%S")
                                                           )

        self.search_dir = os.path.join(self.args.main_path, self.search_dir)

        create_exp_dir(self.search_dir, scripts_to_save=glob.glob('*.py'))

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.search_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger("nao_search")
        self.logger.addHandler(fh)

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
                self.logger.info('seed fixed {}'.format(self.args.seed))

        # NEW PART
        self.child_params = self.get_child_model_params(self.args)
        self.controller_params = self.get_controller_params(self.args)
        self.controller_params['search_dir'] = self.search_dir

        self.corpus = dataloader.Corpus(self.child_params['data_dir'])
        self.eval_batch_size = self.child_params['eval_batch_size']
        self.test_batch_size = 1

        # Tensorboard writer
        self.writer = SummaryWriter(self.search_dir + '/runs/tensor_log_' + self.search_dir)

        self.ntokens = len(self.corpus.dictionary)

        self.best_genotype = None

        self.train_data = batchify(self.corpus.train, self.child_params['batch_size'], self.child_params['cuda'])
        self.val_data = batchify(self.corpus.valid, self.eval_batch_size, self.child_params['cuda'])

        self.child_model = model_search.RNNModelSearch(self.ntokens, self.child_params['emsize'],
                                                       self.child_params['nhid'],
                                                       self.child_params['nhidlast'],
                                                       self.child_params['dropout'], self.child_params['dropouth'],
                                                       self.child_params['dropoutx'],
                                                       self.child_params['dropouti'], self.child_params['dropoute'],
                                                       self.child_params['drop_path'],
                                                       self.child_params['use_avg_leaf'],
                                                       self.child_params['num_intermediate_nodes'],
                                                       self.child_params['handle_hidden_mode'],
                                                       self.child_params['emulate_ws_solutions']
                                                       )

        self.controller_model = controller.Controller(self.controller_params)

        size = 0
        for p in self.child_model.parameters():
            size += p.nelement()
        self.logger.info('child model param size: {}'.format(size))
        size = 0
        for p in self.controller_model.parameters():
            size += p.nelement()
        self.logger.info('controller model param size: {}'.format(size))

        if self.args.cuda:
            if self.args.single_gpu:
                self.parallel_child_model = self.child_model.cuda()
                self.parallel_controller_model = self.controller_model.cuda()
            else:
                self.parallel_child_model = nn.DataParallel(self.child_model, dim=1).cuda()
                self.parallel_controller_model = nn.DataParallel(self.controller_model, dim=1).cuda()
        else:
            self.parallel_child_model = self.child_model
            self.parallel_controller_model = self.controller_model

        total_params = sum(x.data.nelement() for x in self.child_model.parameters())
        self.logger.info('Args: {}'.format(self.args))
        self.logger.info('Child Model total parameters: {}'.format(total_params))
        total_params = sum(x.data.nelement() for x in self.controller_model.parameters())
        self.logger.info('Args: {}'.format(self.args))
        self.logger.info('Controller Model total parameters: {}'.format(total_params))

        self.child_optimizer = torch.optim.SGD(self.child_model.parameters(),
                                               lr=self.child_params['lr'],
                                               weight_decay=self.child_params['wdecay'])
        self.child_epoch = 0

        self.controller_optimizer = torch.optim.Adam(self.controller_model.parameters(),
                                                     lr=self.controller_params['lr'],
                                                     weight_decay=self.controller_params['weight_decay'])
        self.controller_epoch = 0
        self.eval_every_epochs = self.child_params['eval_every_epochs']

    def run(self):

        while True:
            # Train child model
            if self.child_params['arch_pool'] is None:
                '''
                self.arch_pool = generate_arch(self.controller_params['num_seed_arch'],
                                               num_nodes=self.args.num_intermediate_nodes,
                                               num_ops=self.args.num_operations)  # [[arch]]
                self.child_params['arch_pool'] = self.arch_pool
                '''
                arch_set = set()
                arch_list = []
                while len(arch_set) < self.controller_params['num_seed_arch']:
                    new_arch = generate_arch(1,
                                               num_nodes=self.args.num_intermediate_nodes,
                                               num_ops=self.args.num_operations)[0]
                    new_geno = get_genotype(new_arch,
                                            num_nodes=self.args.num_intermediate_nodes,
                                            operations_list=PRIMITIVES,
                                            use_avg_leaf=self.args.use_avg_leaf)
                    new_geno_id = self.search_space.genotype_id_from_geno(genotype=new_geno)
                    if new_geno_id not in arch_set:
                        arch_set.add(new_geno_id)
                        arch_list.append(new_arch)
                        self.logger.info('new arch sampled: {}, geno_id: {}, genotype: {}'.format(new_arch,
                                                                                                  new_geno_id,
                                                                                                  new_geno))
                self.child_params['arch_pool'] = arch_list
                for geno in self.child_params['arch_pool']:
                    self.logger.info('first archs found: {}, using seed {}'.format(geno,
                                                                               self.child_params['seed']))
            self.child_params['arch'] = None

            if isinstance(self.eval_every_epochs, int):
                self.child_params['eval_every_epochs'] = self.eval_every_epochs
            else:
                eval_every_epochs = list(map(int, self.eval_every_epochs))
                for index, e in enumerate(eval_every_epochs):
                    if self.child_epoch < e:
                        self.child_params['eval_every_epochs'] = e
                        break

            for e in range(self.child_params['eval_every_epochs']):
                self.child_epoch += 1
                model_search.train(self.train_data, self.child_model,
                                   self.parallel_child_model, self.child_optimizer,
                                   self.child_params, self.child_epoch)

                if self.child_epoch % self.child_params['eval_every_epochs'] == 0:

                    self.model_path = os.path.join(self.search_dir, 'child')
                    if not os.path.exists(self.model_path):
                        os.mkdir(self.model_path)

                    save_checkpoint(self.child_model,
                                    self.child_optimizer,
                                    self.child_epoch,
                                    self.model_path
                                    )
                    self.logger.info('Saving Model!')

                if self.child_epoch >= self.child_params['train_epochs']:
                    break

            # Evaluate seed archs
            self.valid_accuracy_list = model_search.evaluate(self.val_data, self.child_model,
                                                             self.parallel_child_model, self.child_params,
                                                             self.eval_batch_size)

            # Output archs and evaluated error rate
            old_archs = self.child_params['arch_pool']
            old_archs_perf = self.valid_accuracy_list

            old_archs_sorted_indices = np.argsort(old_archs_perf)
            old_archs = np.array(old_archs)[old_archs_sorted_indices].tolist()
            old_archs_perf = np.array(old_archs_perf)[old_archs_sorted_indices].tolist()
            with open(os.path.join(self.model_path, 'arch_pool.{}'.format(self.child_epoch)), 'w') as fa:
                with open(os.path.join(self.model_path, 'arch_pool.perf.{}'.format(self.child_epoch)), 'w') as fp:
                    with open(os.path.join(self.model_path, 'arch_pool'), 'w') as fa_latest:
                        with open(os.path.join(self.model_path, 'arch_pool.perf'), 'w') as fp_latest:
                            for arch, perf in zip(old_archs, old_archs_perf):
                                genotype = get_genotype(arch,
                                                        num_nodes=self.args.num_intermediate_nodes,
                                                        operations_list=PRIMITIVES,
                                                        use_avg_leaf=self.args.use_avg_leaf)
                                geno_id = self.search_space.genotype_id_from_geno(genotype=genotype)
                                arch = ' '.join(map(str, arch))
                                fa.write('ppl: {}  {}  {},  id: {}\n'.format(math.exp(perf), arch, genotype, geno_id))
                                fa_latest.write('ppl: {}  {}  {},  id: {}\n'.format(math.exp(perf), arch, genotype, geno_id))
                                fp.write('{}\n'.format(perf))
                                fp_latest.write('{}\n'.format(perf))

            self.best_genotype = get_genotype(old_archs[0],
                                              num_nodes=self.args.num_intermediate_nodes,
                                              operations_list=PRIMITIVES,
                                              use_avg_leaf=self.args.use_avg_leaf)

            self.logger.info('BEST ARCH OF EPOCH {} IS: {}'.format(self.child_epoch, self.best_genotype))
            self.logger.info('WITH VALIDATION PPL: {}'.format(old_archs_perf[0]))

            if self.child_epoch >= self.child_params['train_epochs']:
                self.logger.info('Training finished!')
                self.logger.info('BEST ARCH END SEARCH IS: {}'.format(self.best_genotype))
                self.logger.info('WITH VALIDATION PPL: {}'.format(old_archs_perf[0]))

                genotype_id = self.search_space.genotype_id_from_geno(genotype=self.best_genotype)
                return genotype_id, self.best_genotype

            # Train Encoder-Predictor-Decoder
            # [[arch]]
            encoder_input = list(map(lambda x: parse_arch_to_seq(x, num_nodes=self.args.num_intermediate_nodes), old_archs))
            encoder_target = normalize_target(old_archs_perf)
            decoder_target = copy.copy(encoder_input)
            self.controller_params['batches_per_epoch'] = math.ceil(len(encoder_input) / self.controller_params['batch_size'])

            self.controller_epoch = controller.train(encoder_input, encoder_target, decoder_target,
                                                     self.controller_model,
                                                     self.parallel_controller_model,
                                                     self.controller_optimizer,
                                                     self.controller_params,
                                                     self.controller_epoch)

            # Generate new archs
            new_archs = []
            self.controller_params['predict_lambda'] = 0
            top_archs = list(map(lambda x: parse_arch_to_seq(x,
                                                                num_nodes=self.child_params['num_intermediate_nodes']), old_archs[:self.controller_params['top_archs']]))
            max_step_size = self.controller_params['max_step_size']
            while len(new_archs) < self.controller_params['max_new_archs']:
                self.controller_params['predict_lambda'] += 1
                new_arch = controller.infer(top_archs,
                                            self.controller_model,
                                            self.parallel_controller_model,
                                            self.controller_params)
                for arch in new_arch:
                    self.logger.info('arch inferred: {}, type: {}'.format(arch, type(arch)))
                    arch_from_seq = parse_seq_to_arch(arch, num_nodes=self.args.num_intermediate_nodes)
                    self.logger.info('arch from seq: {}'.format(arch_from_seq))
                    infer_geno = get_genotype(parse_seq_to_arch(arch, num_nodes=self.args.num_intermediate_nodes),
                                              num_nodes=self.args.num_intermediate_nodes,
                                              operations_list=PRIMITIVES,
                                              use_avg_leaf=self.args.use_avg_leaf)
                    infer_geno_id = self.search_space.genotype_id_from_geno(genotype=infer_geno)
                    self.logger.info('inferred ARCH geno id: {}, genotype: {}'.format(infer_geno_id, infer_geno))
                    if arch not in encoder_input and arch not in new_archs:
                        new_archs.append(arch)
                        self.logger.info('arch added: {}, type: {}'.format(arch, type(arch)))
                        arch_from_seq = parse_seq_to_arch(arch,num_nodes=self.args.num_intermediate_nodes)
                        self.logger.info('arch from seq: {}'.format(arch_from_seq))
                        infer_geno = get_genotype(parse_seq_to_arch(arch,num_nodes=self.args.num_intermediate_nodes),
                                                num_nodes=self.args.num_intermediate_nodes,
                                                operations_list=PRIMITIVES,
                                                use_avg_leaf=self.args.use_avg_leaf)
                        infer_geno_id = self.search_space.genotype_id_from_geno(genotype=infer_geno)
                        self.logger.info('NEW UNSEEN ARCH FOUND: {}'.format(infer_geno_id))

                    if len(new_archs) >= self.controller_params['max_new_archs']:
                        break
                self.logger.info('{} new archs generated now'.format(len(new_archs)))
                if self.controller_params['predict_lambda'] >= max_step_size:
                    break
            # [[arch]]
            new_archs_seq = new_archs
            new_archs = list(map(lambda x: parse_seq_to_arch(x,num_nodes=self.args.num_intermediate_nodes),
                                 new_archs))  # [[arch]]
            num_new_archs = len(new_archs)
            self.logger.info("Generate {} new archs".format(num_new_archs))
            random_archs = generate_arch(self.controller_params['random_new_archs'],
                                             num_nodes=self.args.num_intermediate_nodes,
                                             num_ops=self.args.num_operations)
            random_new_archs = []
            for rand_arch in random_archs:
                rand_arch_seq = parse_arch_to_seq(rand_arch, num_nodes=self.args.num_intermediate_nodes)
                if rand_arch_seq not in encoder_input and rand_arch_seq not in new_archs_seq:
                    self.logger.info('rand arch seq: {}, from arch: {}'.format(rand_arch_seq, rand_arch))
                    random_new_archs.append(rand_arch)
                    infer_geno = get_genotype(rand_arch,
                                              num_nodes=self.args.num_intermediate_nodes,
                                              operations_list=PRIMITIVES,
                                              use_avg_leaf=self.args.use_avg_leaf)
                    infer_geno_id = self.search_space.genotype_id_from_geno(genotype=infer_geno)
                    self.logger.info('NEW UNSEEN ARCH FOUND from RANDOM: {}'.format(infer_geno_id))

            if len(random_new_archs) > 0:
                new_arch_pool = old_archs[:len(old_archs) - num_new_archs - self.controller_params['random_new_archs']] + new_archs + random_new_archs
            else:
                new_arch_pool = old_archs[:len(old_archs) - num_new_archs] + new_archs

            self.logger.info("Totally {} archs now to train".format(len(new_arch_pool)))
            self.child_params['arch_pool'] = new_arch_pool
            with open(os.path.join(self.model_path, 'arch_pool'), 'w') as f:
                for arch in new_arch_pool:
                    arch = ' '.join(map(str, arch))
                    f.write('{}\n'.format(arch))

    def get_child_model_params(self, args):
        params = {
            'data_dir': args.data_dir,
            'model_dir': os.path.join(args.model_dir, 'child'),
            'arch_pool': args.child_arch_pool,
            'emsize': args.child_emb_size,
            'nhid': args.child_nhid,
            'nhidlast': args.child_emb_size if args.child_nhidlast < 0 else args.child_nhidlast,
            'lr': args.child_lr,
            'clip': args.child_clip,
            'train_epochs': args.child_train_epochs,
            'eval_every_epochs': args.child_eval_every_epochs,
            'batch_size': args.child_batch_size,
            'eval_batch_size': args.child_eval_batch_size,
            'bptt': args.child_bptt,
            'dropout': args.child_dropout,
            'dropouth': args.child_dropouth,
            'dropoutx': args.child_dropoutx,
            'dropouti': args.child_dropouti,
            'dropoute': args.child_dropoute,
            'drop_path': args.child_drop_path,
            'seed': args.seed,
            'nonmono': args.nonmono,
            'log_interval': args.child_log_interval,
            'alpha': args.child_alpha,
            'beta': args.child_beta,
            'wdecay': args.child_weight_decay,
            'small_batch_size': args.child_batch_size if args.child_small_batch_size < 0 else args.child_small_batch_size,
            'max_seq_len_delta': args.child_max_seq_len_delta,
            'cuda': args.cuda,
            'single_gpu': args.single_gpu,
            'gpu': args.gpu,
            'num_intermediate_nodes': args.num_intermediate_nodes,
            'num_operations': args.num_operations,
            'use_avg_leaf' : args.use_avg_leaf,
            'handle_hidden_mode': args.handle_hidden_mode,
            'emulate_ws_solutions': args.emulate_ws_solutions
        }

        return params

    def get_controller_params(self, args):
        params = {
            'model_dir': os.path.join(args.model_dir, 'controller'),
            'shuffle': args.controller_shuffle,
            'num_seed_arch': args.controller_num_seed_arch,
            'encoder_num_layers': args.controller_encoder_num_layers,
            'encoder_hidden_size': args.controller_encoder_hidden_size,
            'encoder_emb_size': args.controller_encoder_emb_size,
            'mlp_num_layers': args.controller_mlp_num_layers,
            'mlp_hidden_size': args.controller_mlp_hidden_size,
            'decoder_num_layers': args.controller_decoder_num_layers,
            'decoder_hidden_size': args.controller_decoder_hidden_size,
            'encoder_dropout': args.controller_encoder_dropout,
            'mlp_dropout': args.controller_mlp_dropout,
            'decoder_dropout': args.controller_decoder_dropout,
            'weight_decay': args.controller_weight_decay,
            'trade_off': args.controller_trade_off,
            'train_epochs': args.controller_train_epochs,
            'save_frequency': args.controller_save_frequency,
            'batch_size': args.controller_batch_size,
            'lr': args.controller_lr,
            'optimizer': args.controller_optimizer,
            'max_gradient_norm': args.controller_max_gradient_norm,
            'predict_beam_width': args.controller_predict_beam_width,
            'predict_lambda': args.controller_predict_lambda,
            'max_step_size': args.controller_max_step_size,
            'max_new_archs': args.controller_max_new_archs,
            'num_intermediate_nodes': args.num_intermediate_nodes,
            'num_operations': args.num_operations,
            'top_archs': args.top_archs,
            'random_new_archs': args.random_new_archs
        }
        return params

