import glob
import math
import os

import IPython
import scipy
import time
import numpy as np
import contextlib
import torch
from torch.autograd import Variable


import utils
from nasws.dataset import dataloader
from . import models
from .tensorboard import TensorBoard
from ..general_search_policies import RNNSearchPolicy
from ..search_space import dag_to_genotype
from utils import create_exp_dir
from . import enas_utils
import torch.nn as nn
import torch.backends.cudnn as cudnn


def _apply_penalties(extra_out, args):
    """Based on `args`, optionally adds regularization penalty terms for
    activation regularization, temporal activation regularization and/or hidden
    state norm stabilization.

    Args:
        extra_out[*]:
            dropped: Post-dropout activations.
            hiddens: All hidden states for a batch of sequences.
            raw: Pre-dropout activations.

    Returns:
        The penalty term associated with all of the enabled regularizations.

    See:
        Regularizing and Optimizing LSTM Language Models (Merity et al., 2017)
        Regularizing RNNs by Stabilizing Activations (Krueger & Memsevic, 2016)
    """
    penalty = 0

    # Activation regularization.
    if args.activation_regularization:
        penalty += (args.activation_regularization_amount *
                    extra_out['dropped'].pow(2).mean())

    # Temporal activation regularization (slowness)
    if args.temporal_activation_regularization:
        raw = extra_out['raw']
        penalty += (args.temporal_activation_regularization_amount *
                    (raw[1:] - raw[:-1]).pow(2).mean())

    # Norm stabilizer regularization
    if args.norm_stabilizer_regularization:
        penalty += (args.norm_stabilizer_regularization_amount *
                    (extra_out['hiddens'].norm(dim=-1) -
                     args.norm_stabilizer_fixed_point).pow(2).mean())

    return penalty


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


def _get_no_grad_ctx_mgr():
    """Returns a the `torch.no_grad` context manager for PyTorch version >=
    0.4, or a no-op context manager otherwise.
    """
    if float(torch.__version__[0:3]) >= 0.4:
        return torch.no_grad()

    return contextlib.suppress()


def _check_abs_max_grad(abs_max_grad, model):
    """Checks `model` for a new largest gradient for this epoch, in order to
    track gradient explosions.
    """
    finite_grads = [p.grad.data
                    for p in model.parameters()
                    if p.grad is not None]

    new_max_grad = max([grad.max() for grad in finite_grads])
    new_min_grad = min([grad.min() for grad in finite_grads])

    new_abs_max_grad = max(new_max_grad, abs(new_min_grad))
    if new_abs_max_grad > abs_max_grad:
        # logger.debug(f'abs max grad {abs_max_grad}')
        return new_abs_max_grad

    return abs_max_grad


class EnasSearchPolicy(RNNSearchPolicy):
    """
    Represent the rnn.Trainer in Pytorch-ENAS.

    """
    def __init__(self, args):
        super(EnasSearchPolicy, self).__init__(args)
        self.initialize_search()

    def initialize_search(self):
        # Put all of the initialization of searching sampler here.

        # Add for filing handling
        search_dir = 'search_{}_ENAS_SEED_{}_{}'.format(self.args.save,
                                                        self.args.seed,
                                                        time.strftime("%Y%m%d-%H%M%S")
                                                        )
        self.search_dir = os.path.join(self.args.main_path, search_dir)
        create_exp_dir(self.search_dir, scripts_to_save=glob.glob('*.py'))
        self.logger = utils.get_logger(name='enas_logger')
        fh = utils.get_file_handler(os.path.join(self.search_dir, 'log.txt'))
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

        # local variable wrapper.
        args = self.args

        self.compute_fisher = False

        # This is the original ENAS part.
        self.controller_step = 0
        self.cuda = args.cuda
        self.dataset = dataloader.Corpus(args.data_path)
        dataset = self.dataset
        self.epoch = 0
        self.shared_step = 0
        self.start_epoch = 0
        self.logger.info("======================================")
        self.logger.info("Initializing ENAS Policy Search... ")
        for regularizer in [('activation regularization',
                             self.args.activation_regularization),
                            ('temporal activation regularization',
                             self.args.temporal_activation_regularization),
                            ('norm stabilizer regularization',
                             self.args.norm_stabilizer_regularization)]:
            if regularizer[1]:
                # self.logger.info('')
                self.logger.info(f'regularizing: {regularizer[0]}')

        self.train_data = enas_utils.batchify(dataset.train,
                                              args.batch_size,
                                              self.cuda)
        # NOTE(brendan): The validation set data is batchified twice
        # separately: once for computing rewards during the Train Controller
        # phase (valid_data, batch size == 64), and once for evaluating ppl
        # over the entire validation set (eval_data, batch size == 1)
        self.valid_data = enas_utils.batchify(dataset.valid,
                                              args.batch_size,
                                              self.cuda)
        self.eval_data = enas_utils.batchify(dataset.valid,
                                             args.test_batch_size,
                                             self.cuda)
        self.test_data = enas_utils.batchify(dataset.test,
                                             args.test_batch_size,
                                             self.cuda)

        self.max_length = self.args.shared_rnn_max_length

        if args.use_tensorboard:
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None
        self.build_model()

        if self.args.load_path:
            self.load_model()

        shared_optimizer = _get_optimizer(self.args.shared_optim)
        controller_optimizer = _get_optimizer(self.args.controller_optim)

        # As fisher information, and it should be seen by this model, to get the loss.

        self.shared_optim = shared_optimizer(
            self.shared.parameters(),
            lr=self.shared_lr,
            weight_decay=self.args.shared_l2_reg)

        self.controller_optim = controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr)

        self.ce = nn.CrossEntropyLoss()

    def build_model(self):
        """Creates and initializes the shared and controller models."""
        if self.args.network_type == 'rnn':
            self.shared = models.RNN(self.args, self.dataset, logger=self.logger)
        else:
            raise NotImplementedError(f'Network type '
                                      f'`{self.args.network_type}` is not '
                                      f'defined')
        self.controller = models.Controller(self.args)

        if self.args.num_gpu == 1:
            self.shared.cuda()
            self.controller.cuda()
        elif self.args.num_gpu > 1:
            raise NotImplementedError('`num_gpu > 1` is in progress')

    def run(self):
        """Cycles through alternately training the shared parameters and the
        controller, as described in Section 2.2, Training ENAS and Deriving
        Architectures, of the paper.

        From the paper (for Penn Treebank):

        - In the first phase, shared parameters omega are trained for 400
          steps, each on a minibatch of 64 examples.

        - In the second phase, the controller's parameters are trained for 2000
          steps.
        """
        #  training logic goes here.

        if self.args.shared_initial_step > 0:
            self.train_shared(self.args.shared_initial_step)
            self.train_controller()

        for self.epoch in range(self.start_epoch, self.args.max_epoch):

            if self.epoch >= self.args.start_using_fisher:
                self.compute_fisher = True

            if self.args.set_fisher_zero_per_iter > 0 \
                    and self.epoch % self.args.set_fisher_zero_per_iter == 0:
                self.shared.set_fisher_zero()

            # 1. Training the shared parameters omega of the child models
            self.train_shared()

            # 2. Training the controller parameters theta
            if self.args.controller_train:
                if self.args.start_training_controller <= self.epoch < self.args.stop_training_controller:
                    self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                with _get_no_grad_ctx_mgr():
                    best_dag = self.derive()
                    _, best_ppl = self.evaluate(self.eval_data,
                                                best_dag,
                                                'val_best',
                                                max_num=self.args.batch_size*100)
                    self.best_genotype = dag_to_genotype(best_dag, num_blocks=self.args.num_blocks)
                    self.best_ppl = best_ppl
                self.save_model()

            if self.epoch >= self.args.shared_decay_after:
                enas_utils.update_lr(self.shared_optim, self.shared_lr)

        #  Finish
        self.logger.info('Training finished!')
        self.logger.info('BEST ARCH END SEARCH IS: {}'.format(self.best_genotype))
        self.logger.info('WITH VALIDATION PPL: {}'.format(self.best_ppl))

        genotype_id = self.search_space.genotype_id_from_geno(genotype=self.best_genotype)
        return genotype_id, self.best_genotype

    def get_loss(self, inputs, targets, hidden, dags):
        """Computes the loss for the same batch for M models.

        This amounts to an estimate of the loss, which is turned into an
        estimate for the gradients of the shared model.

        We store, compute the new WPL.

        """
        if not isinstance(dags, list):
            dags = [dags]

        loss = 0
        for dag in dags:
            output, hidden, extra_out = self.shared(inputs, dag, hidden=hidden)
            output_flat = output.view(-1, self.dataset.num_tokens)
            sample_loss = (self.ce(output_flat, targets) /
                           self.args.shared_num_sample)

            # Get WPL part
            if self.compute_fisher:
                wpl = self.shared.compute_weight_plastic_loss_with_update_fisher(dag)
                wpl = 0.5 * wpl
                loss += sample_loss + wpl
                rest_loss = wpl
            else:
                loss += sample_loss
                rest_loss = Variable(torch.zeros(1))
                # self.logger.info(f'Loss {loss.data[0]} = '
                #             f'sample_loss {sample_loss.data[0]}')

        #assert len(dags) == 1, 'there are multiple `hidden` for multple `dags`'
        return loss, sample_loss, rest_loss, hidden, extra_out

    def train_shared(self, max_step=None):
        """Train the language model for 400 steps of minibatches of 64
        examples.

        Args:
            max_step: Used to run extra training steps as a warm-up.

        BPTT is truncated at 35 timesteps.

        For each weight update, gradients are estimated by sampling M models
        from the fixed controller policy, and averaging their gradients
        computed on a batch of training data.
        """
        valid_ppls=[]
        valid_ppls_after=[]

        model = self.shared
        model.train()
        self.controller.eval()

        hidden = self.shared.init_hidden(self.args.batch_size)
        v_hidden = self.shared.init_hidden(self.args.batch_size)

        if max_step is None:
            max_step = self.args.shared_max_step
        else:
            max_step = min(self.args.shared_max_step, max_step)

        abs_max_grad = 0
        abs_max_hidden_norm = 0
        step = 0
        raw_total_loss = 0
        total_loss = 0
        total_sample_loss = 0
        total_rest_loss = 0
        train_idx = 0
        valid_idx = 0

        def _run_shared_one_batch(inputs, targets, hidden, dags, raw_total_loss):
            # global abs_max_grad
            # global abs_max_hidden_norm
            # global raw_total_loss
            loss, sample_loss, rest_loss, hidden, extra_out = self.get_loss(inputs,
                                                                            targets,
                                                                            hidden,
                                                                            dags)

            # Detach the hidden
            # Because they are input from previous state.
            hidden = enas_utils.detach(hidden)
            raw_total_loss += sample_loss.data / self.args.num_batch_per_iter
            penalty_loss = _apply_penalties(extra_out, self.args)
            loss += penalty_loss
            rest_loss += penalty_loss
            return loss, sample_loss, rest_loss, hidden, extra_out, raw_total_loss

        def _clip_gradient(abs_max_grad, abs_max_hidden_norm):

            h1tohT = extra_out['hiddens']
            new_abs_max_hidden_norm = enas_utils.to_item(
                h1tohT.norm(dim=-1).data.max())
            if new_abs_max_hidden_norm > abs_max_hidden_norm:
                abs_max_hidden_norm = new_abs_max_hidden_norm
                self.logger.debug(f'max hidden {abs_max_hidden_norm}')
            abs_max_grad = _check_abs_max_grad(abs_max_grad, model)
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          self.args.shared_grad_clip)
            return abs_max_grad, abs_max_hidden_norm

        def _evaluate_valid(dag):
            hidden_eval = self.shared.init_hidden(self.args.batch_size)
            inputs_eval, targets_eval = self.get_batch(self.valid_data,
                                                       0,
                                                       self.max_length,
                                                       volatile=True)
            _, valid_loss_eval, _, _, _ = self.get_loss(inputs_eval, targets_eval, hidden_eval, dag)
            valid_loss_eval = enas_utils.to_item(valid_loss_eval.data)
            valid_ppl_eval = math.exp(valid_loss_eval)
            return valid_ppl_eval

        dags_eval = []
        while train_idx < self.train_data.size(0) - 1 - 1:
            if step > max_step:
                break
            dags = self.controller.sample(self.args.shared_num_sample)
            dags_eval.append(dags[0])
            for b in range(0, self.args.num_batch_per_iter):
                # For each model, do the update for 30 batches.
                inputs, targets = self.get_batch(self.train_data,
                                                 train_idx,
                                                 self.max_length)

                loss, sample_loss, rest_loss, hidden, extra_out, raw_total_loss = \
                    _run_shared_one_batch(
                        inputs, targets, hidden, dags, raw_total_loss)

                # update with complete logic
                # First, normally we compute one loss and do update accordingly.
                # if in the last batch, we compute the fisher information
                # based on two kinds of loss, complete or ce-loss only.
                self.shared_optim.zero_grad()

                # If it is the last training batch. Update the Fisher information
                if self.compute_fisher and (not self.args.shared_valid_fisher):
                    if b == self.args.num_batch_per_iter - 1:
                        sample_loss.backward()
                        if self.args.shared_ce_fisher:
                            self.shared.update_fisher(dags[0])
                            rest_loss.backward()
                        else:
                            rest_loss.backward()
                            self.shared.update_fisher(dags[0])
                    else:
                        loss.backward()
                else:
                    loss.backward()

                abs_max_grad, abs_max_hidden_norm = _clip_gradient(abs_max_grad, abs_max_hidden_norm)

                self.shared_optim.step()

                total_loss += loss.data / self.args.num_batch_per_iter
                total_sample_loss += sample_loss.data / self.args.num_batch_per_iter
                total_rest_loss += rest_loss.data / self.args.num_batch_per_iter

                train_idx = ((train_idx + self.max_length) %
                             (self.train_data.size(0) - 1))

            if self.epoch > self.args.start_evaluate_diff:
                valid_ppl_eval = _evaluate_valid(dags[0])
                valid_ppls.append(valid_ppl_eval)

            self.logger.debug(f'Step {step}'
                              f'Loss {enas_utils.to_item(total_loss) / (step + 1)} = '
                              f'sample_loss {enas_utils.to_item(total_sample_loss) / (step + 1)} + '
                              f'wpl {enas_utils.to_item(total_rest_loss) / (step + 1)}')

            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_shared_train(total_loss, raw_total_loss)
                raw_total_loss = 0
                total_loss = 0
                total_sample_loss = 0
                total_rest_loss = 0

            if self.compute_fisher:
                # Update with the validation dataset for fisher information after each step,
                # with update the optimal weights.
                v_inputs, v_targets = self.get_batch(self.valid_data,
                                                     valid_idx,
                                                     self.max_length)
                v_loss, v_sample_loss, _, v_hidden, v_extra_out, _ = _run_shared_one_batch(
                    v_inputs, v_targets, v_hidden, dags, 0)
                self.shared_optim.zero_grad()
                if self.args.shared_ce_fisher:
                    v_sample_loss.backward()
                else:
                    v_loss.backward()
                self.shared.update_fisher(dags[0], self.epoch)
                self.shared.update_optimal_weights()
                valid_idx = ((valid_idx + self.max_length) %
                             (self.valid_data.size(0) - 1))

            step += 1
            self.shared_step += 1

        if self.epoch > self.args.start_evaluate_diff:
            for arch in dags_eval:
                valid_ppl_eval = _evaluate_valid(arch)
                valid_ppls_after.append(valid_ppl_eval)
                self.logger.debug(f'valid_ppl {valid_ppl_eval}')
            diff = np.array(valid_ppls_after) - np.array(valid_ppls)
            self.logger.info(f'Mean_diff {np.mean(diff)}')
            self.logger.info(f'Max_diff {np.amax(diff)}')
            self.tb.scalar_summary(f'Mean difference', np.mean(diff), self.epoch)
            self.tb.scalar_summary(f'Max difference', np.amax(diff), self.epoch)
            self.tb.scalar_summary(f'Mean valid_ppl after training', np.mean(np.array(valid_ppls_after)), self.epoch)
            self.tb.scalar_summary(f'Mean valid_ppl before training', np.mean(np.array(valid_ppls)), self.epoch)
            self.tb.scalar_summary(f'std_diff', np.std(np.array(diff)), self.epoch)

    def get_reward(self, dag, entropies, hidden, valid_idx=None):
        """
        Computes the perplexity of a single sampled model on a minibatch of
        validation data.

        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()

        if valid_idx is None:
            valid_idx = 0

        inputs, targets = self.get_batch(self.valid_data,
                                         valid_idx,
                                         self.max_length,
                                         volatile=True)
        _, valid_loss, _, hidden, _ = self.get_loss(inputs, targets, hidden, dag)
        valid_loss = enas_utils.to_item(valid_loss.data)

        valid_ppl = math.exp(valid_loss)


        # TODO: we don't know reward_c
        if self.args.ppl_square:
            # TODO: but we do know reward_c=80 in the previous paper
            R = self.args.reward_c / valid_ppl ** 2
        else:
            R = self.args.reward_c / valid_ppl

        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden

    def train_controller(self):
        """Fixes the shared parameters and updates the controller parameters.

        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being c/valid_ppl, where valid_ppl
        is computed on a minibatch of validation data.

        A moving average baseline is used.

        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -> second (Train Controller) phase).
        """
        model = self.controller
        model.train()
        # TODO(brendan): Why can't we call shared.eval() here? Leads to loss
        # being uniformly zero for the controller.
        # self.shared.eval()

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.shared.init_hidden(self.args.batch_size)
        total_loss = 0
        valid_idx = 0
        for step in range(self.args.controller_max_step):
            # print("************ train controller ****************")
            # sample models
            dags, log_probs, entropies = self.controller.sample(
                batch_size=self.args.policy_batch_size, with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            # NOTE(brendan): No gradients should be backpropagated to the
            # shared model during controller training, obviously.
            with _get_no_grad_ctx_mgr():
                rewards, hidden = self.get_reward(dags,
                                                  np_entropies,
                                                  hidden,
                                                  valid_idx)

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss
            loss = -log_probs * enas_utils.get_variable(adv,
                                                        self.cuda,
                                                        requires_grad=False)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += enas_utils.to_item(loss.data)

            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_controller_train(total_loss,
                                                 adv_history,
                                                 entropy_history,
                                                 reward_history,
                                                 avg_reward_base,
                                                 dags)

                reward_history, adv_history, entropy_history = [], [], []
                total_loss = 0

            self.controller_step += 1

            prev_valid_idx = valid_idx
            valid_idx = ((valid_idx + self.max_length) %
                         (self.valid_data.size(0) - 1))
            # NOTE(brendan): Whenever we wrap around to the beginning of the
            # validation data, we reset the hidden states.
            if prev_valid_idx > valid_idx:
                hidden = self.shared.init_hidden(self.args.batch_size)

    def evaluate(self, source, dag, name, batch_size=1, max_num=None):
        """Evaluate on the validation set.

        NOTE(brendan): We should not be using the test set to develop the
        algorithm (basic machine learning good practices).
        """
        self.shared.eval()
        self.controller.eval()

        data = source[:max_num*self.max_length]

        total_loss = 0
        hidden = self.shared.init_hidden(batch_size)

        pbar = range(0, data.size(0) - 1, self.max_length)
        for count, idx in enumerate(pbar):
            inputs, targets = self.get_batch(data, idx, volatile=True)
            output, hidden, _ = self.shared(inputs,
                                            dag,
                                            hidden=hidden,
                                            is_train=False)
            output_flat = output.view(-1, self.dataset.num_tokens)
            total_loss += len(inputs) * self.ce(output_flat, targets).data
            hidden = enas_utils.detach(hidden)
            ppl = math.exp(enas_utils.to_item(total_loss) / (count + 1) / self.max_length)

        val_loss = enas_utils.to_item(total_loss) / len(data)
        ppl = math.exp(val_loss)

        self.tb.scalar_summary(f'eval/{name}_loss', val_loss, self.epoch)
        self.tb.scalar_summary(f'eval/{name}_ppl', ppl, self.epoch)
        self.logger.info(f'eval | loss: {val_loss:8.2f} | ppl: {ppl:8.2f}')
        return val_loss, ppl

    def derive(self, sample_num=None, valid_idx=0):
        """TODO(brendan): We are always deriving based on the very first batch
        of validation data? This seems wrong...
        """
        hidden = self.shared.init_hidden(self.args.batch_size)

        if sample_num is None:
            sample_num = self.args.derive_num_sample

        dags, _, entropies = self.controller.sample(sample_num,
                                                    with_details=True)

        max_R = 0
        best_dag = None
        for dag in dags:
            R, _ = self.get_reward(dag, entropies, hidden, valid_idx)
            if R.max() > max_R:
                max_R = R.max()
                best_dag = dag

        self.logger.info(f'derive | max_R: {max_R:8.6f}')
        fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                 f'{max_R:6.4f}-best.png')
        path = os.path.join(self.args.model_dir, 'networks', fname)
        res = enas_utils.draw_network(best_dag, path)
        if res:
            self.tb.image_summary('derive/best', [path], self.epoch)

        return best_dag

    @property
    def shared_lr(self):
        degree = max(self.epoch - self.args.shared_decay_after + 1, 0)
        return self.args.shared_lr * (self.args.shared_decay ** degree)

    @property
    def controller_lr(self):
        return self.args.controller_lr

    def get_batch(self, source, idx, length=None, volatile=False):
        # code from
        # https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        length = min(length if length else self.max_length,
                     len(source) - 1 - idx)
        data = Variable(source[idx:idx + length], volatile=volatile)
        target = Variable(source[idx + 1:idx + 1 + length].view(-1),
                          volatile=volatile)
        return data, target

    @property
    def shared_path(self):
        return f'{self.args.model_dir}/shared_epoch{self.epoch}_step{self.shared_step}.pth'

    @property
    def controller_path(self):
        return f'{self.args.model_dir}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.model_dir, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):
        torch.save(self.shared.state_dict(), self.shared_path)
        self.logger.info(f'[*] SAVED: {self.shared_path}')

        torch.save(self.controller.state_dict(), self.controller_path)
        self.logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.model_dir, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                enas_utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            self.logger.info(f'[!] No checkpoint found in {self.args.model_dir}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.shared_step = max(shared_steps)
        self.controller_step = max(controller_steps)

        if self.args.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        self.shared.load_state_dict(
            torch.load(self.shared_path, map_location=map_location))
        self.logger.info(f'[*] LOADED: {self.shared_path}')

        self.controller.load_state_dict(
            torch.load(self.controller_path, map_location=map_location))
        self.logger.info(f'[*] LOADED: {self.controller_path}')
        # IPython.embed()

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    reward_history,
                                    avg_reward_base,
                                    dags):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / self.args.log_step

        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward

        self.logger.info(
            f'training controller | epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('controller/loss',
                                   cur_loss,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward',
                                   avg_reward,
                                   self.controller_step)
            self.tb.scalar_summary('controller/std/reward',
                                   np.std(reward_history),
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward-B_per_epoch',
                                   avg_reward - avg_reward_base,
                                   self.controller_step)
            self.tb.scalar_summary('controller/entropy',
                                   avg_entropy,
                                   self.controller_step)
            self.tb.scalar_summary('controller/adv',
                                   avg_adv,
                                   self.controller_step)

            paths = []
            res = False
            for dag in dags:
                fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                         f'{avg_reward:6.4f}.png')
                path = os.path.join(self.args.model_dir, 'networks', fname)
                res = enas_utils.draw_network(dag, path)
                paths.append(path)
            if res:
                self.tb.image_summary('controller/sample',
                                      paths,
                                      self.controller_step)

    def _summarize_shared_train(self, total_loss, raw_total_loss):
        """Logs a set of training steps."""
        cur_loss = enas_utils.to_item(total_loss) / self.args.log_step
        cur_raw_loss = enas_utils.to_item(raw_total_loss) / self.args.log_step
        try:
            ppl = math.exp(cur_raw_loss)
        except RuntimeError as e:
            print(f"Got error {e}")
            IPython.embed()

        self.logger.info(f'training shared |'
                         f'| epoch {self.epoch:3d} '
                         f'| lr {self.shared_lr:4.2f} '
                         f'| raw loss {cur_raw_loss:.2f} '
                         f'| loss {cur_loss:.2f} '
                         f'| ppl {ppl:8.2f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('shared/loss',
                                   cur_loss,
                                   self.shared_step)
            self.tb.scalar_summary('shared/perplexity',
                                   ppl,
                                   self.shared_step)
