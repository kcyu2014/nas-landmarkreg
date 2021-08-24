# define the search api here with the help of NNI
import json
import os 
from collections import OrderedDict
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monodepth.utils.sync_batchnorm import convert_model, patch_replication_callback
from monodepth.utils.reporter import Reporter
from monodepth.utils.checkpointer import Checkpointer, save_json, load_json
# from monodepth.utils.checkpointer
from nni.nas.pytorch.base_trainer import BaseTrainer
from nni.nas.pytorch.trainer import TorchTensorEncoder
from nni.nas.pytorch.spos import SPOSSupernetTrainer
from nni.nas.pytorch.random import RandomMutator
from nni.nas.pytorch.mutator import Mutator
from nni.nas.pytorch.mutables import MutableScope, LayerChoice, InputChoice
from .constants import PRIMITIVES, NONE
from .landmark_procedure import SearchSpace, landmark_loss_step_fns
from .nasbench_monodepth import query_nb201_trial_stats, Nb201TrialConfig

from .utils import LandmarkLossAverageMeter
import IPython
import logging


def arch_to_mutator_cache(arch, mutables):
    """From Json Arch to cache that can replace mutator._cache

    Parameters
    ----------
    arch : dict() 'str' -> 'int'
        Json type string, the format saved to disk
    mutables : mutator.mutables
        Mutable objects in mutator

    Returns
    -------
    dict(torch.one_hot tensors)
        To replace the mutator._cache as a fix type architecture

    Raises
    ------
    NotImplementedError
        When calling InputChoice, not yet supported...
    """
    result = dict()
    # for mutable in self.mutables:
    for mutable in mutables:
        if isinstance(mutable, LayerChoice):
            gen_index = torch.Tensor([arch[mutable.key]]).to(torch.int64).view(1, -1)
            result[mutable.key] = F.one_hot(gen_index, num_classes=len(mutable)).view(-1).bool()
        elif isinstance(mutable, InputChoice):
            raise NotImplementedError('Not yet support assign InputChoice, study later...')
            if mutable.n_chosen is None:
                result[mutable.key] = torch.randint(high=2, size=(mutable.n_candidates,)).view(-1).bool()
            else:
                perm = torch.randperm(mutable.n_candidates)
                mask = [i in perm[:mutable.n_chosen] for i in range(mutable.n_candidates)]
                result[mutable.key] = torch.tensor(mask, dtype=torch.bool)  # pylint: disable=not-callable
    return result

def assign_arch_to_mutator(mutator, arch):
    """Assign the mutator corresponding architecture cache.


    Parameters
    ----------
    mutator : [type]
        [description]
    arch : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    NotImplementedError
        [description]
    RuntimeError
        [description]
    """
    mutator._cache = arch_to_mutator_cache(arch, mutator.mutables)
    return mutator


class NonZeroRandomMutatorSync(RandomMutator):
    """NonZeroRandomMutator with synchronization all the cell structures.
    
    Logical flow
        Each cell is defined as a scope
        Reset the _cell_choices after each new call `sample_search` 
            If the _cell_choices is None, 
                Sample the cell as usual
                Store the sampled arch into a dict
            else:
                query the saved _cell_choices
                overwrite directly instead of sampling again.
        
    """
    ZERO_INDEX = PRIMITIVES.index(NONE)

    def __init__(self, model):
        super().__init__(model)        
        self.reset()
        self._choices = None

    def _sample_one_cell(self, scope):
        if self._cell_choices is None:
            self._cell_choices = dict()
            while True:
                has_non_zero = False
                for mutable in scope.modules():
                    if isinstance(mutable, LayerChoice):
                        gen_index = torch.randint(high=len(mutable), size=(1, ))
                        choice = F.one_hot(gen_index, num_classes=len(mutable)).view(-1).bool()
                        self._choices[mutable.key] = choice
                        if not has_non_zero: # overwrite has_non_zero if False, otherwise keep it True
                            has_non_zero = not choice[self.ZERO_INDEX]
                
                if has_non_zero:
                    break
            # process the cell_choices:
            for k, v in self._choices.items():
                self._cell_choices[k.split('_')[1]] = v.clone()
        else:
            # assign to the same cell_choices to sync all scope.
            for mutable in scope.modules():
                if isinstance(mutable, LayerChoice):
                    self._choices[mutable.key] = self._cell_choices[mutable.key.split('_')[1]].clone()
            
    def sample_search(self):
        self._choices = dict()
        self._cell_choices = None
        for mutable in self.mutables:
            if isinstance(mutable, MutableScope):
                # use MutableScope directly, 
                self._sample_one_cell(mutable)
            
            # elif isinstance(mutable, InputChoice):
            #     if mutable.n_chosen is None:
            #         result[mutable.key] = torch.randint(high=2, size=(mutable.n_candidates,)).view(-1).bool()
            #     else:
            #         perm = torch.randperm(mutable.n_candidates)
            #         mask = [i in perm[:mutable.n_chosen] for i in range(mutable.n_candidates)]
            #         result[mutable.key] = torch.tensor(mask, dtype=torch.bool)  # pylint: disable=not-callable
        # print('sampled one arch')
        return self._choices


class NonZeroRandomMutator(RandomMutator):
    ZERO_INDEX = PRIMITIVES.index(NONE)

    def __init__(self, model):
        super().__init__(model)        
        self.reset()
        # import IPython
        # IPython.embed()
        self._choices = None

    def _sample_one_cell(self, scope):
        while True:
            has_non_zero = False
            for mutable in scope.modules():
                if isinstance(mutable, LayerChoice):
                    gen_index = torch.randint(high=len(mutable), size=(1, ))
                    choice = F.one_hot(gen_index, num_classes=len(mutable)).view(-1).bool()
                    self._choices[mutable.key] = choice
                    if not has_non_zero: # overwrite has_non_zero if False, otherwise keep it True
                        has_non_zero = not choice[self.ZERO_INDEX]
            
            if has_non_zero:
                break

    def sample_search(self):
        self._choices = dict()
        for mutable in self.mutables:
            if isinstance(mutable, MutableScope):
                # use MutableScope directly, 
                self._sample_one_cell(mutable)
            
        return self._choices


class MonoDepthNASTraining(BaseTrainer):
    
    def __init__(self, 
                    model, 
                    mutator:Mutator, 
                    tasks,
                    optimizer,
                    scheduler,
                    epoch_length,
                    num_epochs,
                    dataloader,
                    device,
                    sync_bn=False,
                    validation=None,
                    checkpointer:Checkpointer=None,
                    reporter:Reporter=None, 
                    args=None, 
                    callbacks=None): 
        """ Monocular Depth Estimation NAS Trainer.


        Parameters
        ----------
        model : model with mutables and layer choices
            Model Search with One Shot approach.
        mutator : Mutator class in `nni.nas.pytorch`
            Defines the NAS sampling algorithm
        tasks : Tasks in MonoDepth
            Original in params.
        optimizer : torch.optim.Optimizer
            Original
        scheduler : torch scheduler for learning rate
            torch scheduler for learning rate
        epoch_length : int  
            Epoch length
        num_epochs : int
            Total epoch number
        dataloader : dataloader
            As original, do not modify
        device : str
            target device
        sync_bn : bool, optional
            Using sync-bn or not, by default False
        validation : original, optional
            As original, do not modify, by default None
        checkpointer : utils.Checkpointer, optional
            As original, do not modify, by default None
        reporter : utils.Reporter, optional
            As original, do not modify, by default None
        args : Argparse, optional
            stores the NAS args, by default None
        callbacks : List[nni.pytorch.nas.callbacks], optional
            Callbacks NNI, executed before end of each epoch, by default None
        """

        super().__init__()
        self.model_cpu = model

        self.tasks = tasks
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch_length = epoch_length
        self.num_epochs = num_epochs

        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.sync_bn = sync_bn

        self.validation = validation
        self.checkpointer = checkpointer
        self.reporter = reporter

        self.model = None # do not set any counter here.
        self.mutator = mutator
        self.optimizer = optimizer

        self.num_epochs = num_epochs
        
        self.log_dir = reporter.log_dir
        
        self.status_writer = open(os.path.join(self.log_dir, "nni-log"), "w")
        self.callbacks = callbacks if callbacks is not None else []
        for callback in self.callbacks:
            callback.build(self.model, self.mutator, self)
        self.args = args

        # init the landmark training
        if args.supernet_train_method == 'spos_rankloss': 
            self.construct_landmark_search_space()
            self.landmark_step_fn = landmark_loss_step_fns[args.landmark_loss_procedure]
            self.landmark_loss_obj = LandmarkLossAverageMeter()
        else:
            self.landmark_step_fn = None
            self.landmark_loss_obj = None
        
    def train_one_epoch(self, epoch):
        """
        Train one epoch.

        Parameters
        ----------
        epoch : int
            Epoch number starting from 0.
        """
        self.args.tmp_epoch = epoch
        self.reporter.write_epoch_start(epoch, self.num_epochs)

        # init epoch
        num_images_total = 0
        num_batches_each = 0
        cumulative_losses = OrderedDict(
            {d: 0 for d in self.dataloader.dataset_names}
        )
        cumulative_loss = 0

        # train epoch
        while num_images_total < self.epoch_length:
            batches = next(self.dataloader)

            num_batches_each += 1

            # input
            for batch in batches.values():
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                # print(len(batch["image"]), self.epoch_length)
                # exit()
                num_images_total += len(batch["image"])

            def model_forward_fn(model, batch, dataset):
                """ Model forward with a given batch and current dataset name """ 
                res = model(batch["image"])  # , batch["candidates"])
                
                if isinstance(res, tuple):  # len(res) > 1:
                    _loss = 0
                    for pred in res:
                        _loss += self.parallel_tasks[dataset]["loss"](
                            pred, batch["disparity"], batch["mask"]
                        )

                    _loss /= len(res)
                    display_1 = res[0]
                    display_2 = res[1]
                else:
                    prediction = res
                    _loss = self.parallel_tasks[dataset]["loss"](
                        prediction, batch["disparity"], batch["mask"]
                    )
                    
                    display_1 = prediction
                    display_2 = prediction
                
                return _loss, display_1, display_2

            # reset gradients, forward pass, loss, backward pass
            def closure(weights, report=False):
                """ Compute the closure """
                self.optimizer.zero_grad()
                self.mutator.reset()
                arch_cache = self.mutator._cache
                loss = 0
                losses = {}

                for i, (dataset, batch) in enumerate(batches.items()):
                    
                    if weights[i] != 0:
                        losses[i], display_1, display_2 = model_forward_fn(self.model, batch, dataset)
                        _loss = weights[i] * losses[i]
                        _loss.backward()
                        loss += _loss.item()
                        
                        if report and num_images_total == self.epoch_length:
                            self.reporter.plot(
                                dataset, batch, display_1, display_2
                            )
                        # using landmark loss here
                        if self.args.supernet_train_method == 'spos_rankloss':
                            # add the landmark procedure here
                            self.landmark_step_fn(self.model, batch, self.landmark_search_space, dataset, 
                                    self.args, lambda m, a: assign_arch_to_mutator(self.mutator, a), model_forward_fn, self.landmark_loss_obj)
                            self.mutator._cache = arch_cache

                torch.cuda.empty_cache()
                # assign the same cache back
                self.mutator._cache = arch_cache
                return loss, losses

            # optimize
            loss, losses = self.optimizer.step(closure)

            cumulative_loss += loss

            for i, d in enumerate(batches):
                cumulative_losses[d] += losses[i]

            # verbose
            self.reporter.write_train(
                loss,
                cumulative_loss,
                num_batches_each,
                num_images_total,
                self.epoch_length,
                self.landmark_loss_obj
            )
            if self.args.debug and num_images_total > 200:
                self.reporter.info('Break the training due to debugging...')
                break
        
        # report
        self.reporter.write_epoch_end(
            epoch, cumulative_loss, cumulative_losses, num_batches_each, self.landmark_loss_obj
        )

    def validate_one_epoch(self, epoch):
        """
        Validate one epoch.

        Parameters
        ----------
        epoch : int
            Epoch number starting from 0.
        """
                    # validation
        try:
            self.validation.validate_with_mutator(self.mutator, epoch)
            if epoch % self.args.save_every_epoch == 0 and epoch > 0:
                self.validate_landmarks(epoch)
        except RuntimeError as e:
            print("Something went wrong during validation (probably OOM)")
    
    def model_to_gpus(self):
        if self.sync_bn:
            print("Using syncronized batchnorm")
            self.model = nn.DataParallel(convert_model(self.model_cpu))
            patch_replication_callback(model)
            self.model.to(self.device)
        else:
            # print("Using standard batchnorm")
            print("Using the modified data layer")
            # use Customized DataParallel and criterion
            from monodepth.utils.parallel import DataParallelModel, DataParallelCriterion

            self.parallel_tasks = self.tasks
            # self.parallel_tasks = OrderedDict()
            # for dataset, config in self.tasks.items():
            #     self.parallel_tasks[dataset] = OrderedDict()
            #     for c_name, c in config.items():
            #         if c_name == 'loss':
            #             c = DataParallelCriterion(c)
            #         self.parallel_tasks[dataset][c_name] = c
            self.model = nn.DataParallel(self.model_cpu)
            # self.model = DataParallelModel(self.model_cpu)
            self.model.to(self.device)
        
        # self.model = model
        self.mutator.to(self.device)
        self.optimizer.to(self.device)

    def train(self):
        """
        Train ``num_epochs``.
        Trigger callbacks at the start and the end of each epoch.

        Parameters
        ----------
        validate : bool
            If ``true``, will do validation every epoch.
        """
        start_epoch = self.checkpointer.load()
        self.model_to_gpus()

        for epoch in range(start_epoch, self.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # training
            self.reporter.info(f"Epoch {epoch + 1} Training")
            torch.cuda.empty_cache()
            self.train_one_epoch(epoch)
            torch.cuda.empty_cache()
            if self.validation:
                # validation
                self.reporter.info(f"Epoch {epoch + 1} Validating")
                self.validate_one_epoch(epoch) 

            self.checkpointer.save(epoch)
            self.scheduler.step()

            for callback in self.callbacks:
                callback.on_epoch_end(epoch)

        return self.model_cpu, self.reporter.cumulative_loss

    def validate(self):
        """
        Do one validation.
        """
        self.validate_one_epoch(-1)

    def export(self, file):
        """
        Call ``mutator.export()`` and dump the architecture to ``file``.

        Parameters
        ----------
        file : str
            A file path. Expected to be a JSON.
        """
        mutator_export = self.mutator.export()
        with open(file, "w") as f:
            json.dump(mutator_export, f, indent=2, sort_keys=True, cls=TorchTensorEncoder)

    def checkpoint(self):
        """
        Return trainer checkpoint.
        """
        raise NotImplementedError("Not implemented yet")

    def enable_visualization(self):
        """
        Enable visualization. Write graph and training log to folder ``logs/<timestamp>``.
        """
        sample = None
        for x, _ in self.train_loader:
            sample = x.to(self.device)[:2]
            break
        if sample is None:
            self.reporter.info("Sample is %s.", sample)
        self.reporter.info("Creating graph json, writing to %s. Visualization enabled.", self.log_dir)
        with open(os.path.join(self.log_dir, "graph.json"), "w") as f:
            json.dump(self.mutator.graph(sample), f)
        self.visualization_enabled = True

    def _write_graph_status(self):
        if hasattr(self, "visualization_enabled") and self.visualization_enabled:
            self.reporter.info(json.dumps(self.mutator.status()), file=self.status_writer, flush=True)

    def test_oneshot(self, epoch):
        # these are the total steps.
        if self.args.evaluate_sampler == 'random':
            # using random muator is fine.
            archs = []
            losses = []
            for i in range(self.args.evaluate_step_budget):
                self.mutator.reset()
                # validation loss
                res = self.validation.validate(epoch, do_report=False)
                arch = self.mutator.export()
                val_loss = res['ReDWeb'].item()
                archs.append(arch)
                losses.append(val_loss)
    
            TOP_K = 10
            top_k_indices = np.argsort(losses)[:TOP_K]
            top_k_archs = [archs[i] for i in top_k_indices]
            top_k_losses = [losses[i] for i in top_k_indices]
            
            self.checkpointer.save_nas_final_result(top_k_archs, top_k_losses, epoch)
            self.reporter.info(f'Saving top {TOP_K} architectures to files using random search.')
            self.reporter.info(f'{top_k_archs}')
        else:
            raise NotImplementedError('Not supported for other evaluate_sampler at this moment.')
        
        return archs, losses
    
    def get_search_space_architectures(self, ignore_landmark=True):
        """Obtain the landmark architectures

        Returns
        -------
        [type]
            [description]
        """
        query_conditions = [
            Nb201TrialConfig.mode == 'sync' if 'sync' in self.args.mutator else Nb201TrialConfig.mode == 'normal', 
            Nb201TrialConfig.space == self.args.search_space.replace('_', '-')
        ]

        if ignore_landmark:
            res = []
            for d in query_nb201_trial_stats(None, None, 'redweb', other_conditions=query_conditions):
                if eval(d['config']['arch']) not in self.landmark_search_space._landmark_specs:
                    res.append(d)
        else:
            res = list(query_nb201_trial_stats(None, None, 'redweb', other_conditions=query_conditions))
        return res

    def construct_landmark_search_space(self):
        num_landmarks = self.args.landmark_num_archs
        specs_weights = []

        if self.args.landmark_sample_method == 'fixed':
            raise NotImplementedError('Fixed sampling is disabled as landmarks too small')
            # this will sample the
        elif self.args.landmark_sample_method == 'random':
            # will just support the first architecture
            # return the first number landmarks as the architecture are random sampled in the beginning
            for ind, d in enumerate(self.get_search_space_architectures(False)):
                if ind >= num_landmarks:
                    break
                arch = eval(d['config']['arch'])
                val_loss = d['valid_loss']
                specs_weights.append((arch, val_loss))
        else:
            raise NotImplementedError('not yet supported')
        specs_weights.sort(key=lambda x: x[1], reverse=True)
        # IPython.embed(header='checking construction of landmark space')
        specs, weights = zip(*specs_weights)
        ids = list(range(num_landmarks))
        self.landmark_search_space = SearchSpace(ids, specs, weights)

    def validate_landmarks(self, epoch, debug=False):
        # just validate the landmark architectures and get results of statistics
        # translate the script to node
        """Here, we get the landmark architectures by the simple loop.

            # under certain search space mode
        """

        archs = []
        gt_losses = []
        pred_losses = []
        original_arch = self.mutator._cache
        # IPython.embed(header='checking landmark query issues...')
        # for d in query_nb201_trial_stats(None, None, None, other_conditions=query_conditions):
        count = 0
        for ind, d in enumerate(self.get_search_space_architectures()):
            # each of this is one landmark architecture
            arch = eval(d['config']['arch'])
            val_loss = d['valid_loss']
            count += 1
            if debug:
                continue
            # get the validation loss, i.e. append the given architecture to mutator
            assign_arch_to_mutator(self.mutator, arch)
            # validate here if the export return the correct cache
            res = self.validation.validate(epoch, do_report=False, force=True)
            
            try:
                val_loss_pred = res['ReDWeb'].item()
            except Exception as e: 
                self.reporter.info(f'WARNING: Error met in architecture {arch}, {e}')
                IPython.embed()
                continue
            # append same here
            archs.append(arch)
            gt_losses.append(val_loss)
            pred_losses.append(val_loss_pred)
        
        if debug:
            self.reporter.info(f'Total kdt number to compute landmark {count}')
            if count == 0:
                raise RuntimeError('No KdT landmarks! Fatal error!!')
        
        if count > 0:
        # compute the kendall tau
            self.reporter.write_kendall_tau(archs, gt_losses, pred_losses, epoch)
        else:
            raise RuntimeError('No KdT landmarks computed, fatal error!!!')
        # assign the original architecture to the mutator
        self.mutator._cache = original_arch
