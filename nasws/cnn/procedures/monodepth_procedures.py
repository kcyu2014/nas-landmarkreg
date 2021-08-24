"""
Put the monodepth related procedures here!
This is just in case. ideally we should not even do this, but compute the loss only in the model_forward_fn

"""
from collections import OrderedDict
import torch
import logging


def monodepth_train_one_epoch(policy, epoch):
    policy.args.tmp_epoch = epoch
    policy.reporter.write_epoch_start(epoch, policy.num_epochs)

    # init epoch
    num_images_total = 0
    num_batches_each = 0
    cumulative_losses = OrderedDict(
        {d: 0 for d in policy.dataloader.dataset_names}
    )
    cumulative_loss = 0

    # train epoch
    while num_images_total < policy.epoch_length:
        batches = next(policy.dataloader)

        num_batches_each += 1

        # input
        for batch in batches.values():
            for k, v in batch.items():
                batch[k] = v.to(policy.device)
            # print(len(batch["image"]), policy.epoch_length)
            # exit()
            num_images_total += len(batch["image"])

        def model_forward_fn(model, batch, dataset):
            """ Model forward with a given batch and current dataset name """ 
            res = model(batch["image"])  # , batch["candidates"])
            
            if isinstance(res, tuple):  # len(res) > 1:
                _loss = 0
                for pred in res:
                    _loss += policy.parallel_tasks[dataset]["loss"](
                        pred, batch["disparity"], batch["mask"]
                    )

                _loss /= len(res)
                display_1 = res[0]
                display_2 = res[1]
            else:
                prediction = res
                _loss = policy.parallel_tasks[dataset]["loss"](
                    prediction, batch["disparity"], batch["mask"]
                )
                
                display_1 = prediction
                display_2 = prediction
            
            return _loss, display_1, display_2

        # reset gradients, forward pass, loss, backward pass
        def closure(weights, report=False):
            """ Compute the closure """
            policy.optimizer.zero_grad()
            policy.mutator.reset()
            arch_cache = policy.mutator._cache
            loss = 0
            losses = {}

            for i, (dataset, batch) in enumerate(batches.items()):
                
                if weights[i] != 0:
                    losses[i], display_1, display_2 = model_forward_fn(policy.model, batch, dataset)
                    _loss = weights[i] * losses[i]
                    _loss.backward()
                    loss += _loss.item()
                    
                    if report and num_images_total == policy.epoch_length:
                        policy.reporter.plot(
                            dataset, batch, display_1, display_2
                        )
                    # using landmark loss here
                    if policy.args.supernet_train_method == 'spos_rankloss':
                        # add the landmark procedure here
                        logging.debug('compute landmark loss')
                        policy.landmark_step_fn(policy.model, batch, policy.landmark_search_space, dataset, 
                                policy.args, lambda m, a: assign_arch_to_mutator(policy.mutator, a), model_forward_fn, policy.landmark_loss_obj)
                        policy.mutator._cache = arch_cache

            torch.cuda.empty_cache()
            # assign the same cache back
            policy.mutator._cache = arch_cache
            return loss, losses

        # optimize
        loss, losses = policy.optimizer.step(closure)

        cumulative_loss += loss

        for i, d in enumerate(batches):
            cumulative_losses[d] += losses[i]

        # verbose
        policy.reporter.write_train(
            loss,
            cumulative_loss,
            num_batches_each,
            num_images_total,
            policy.epoch_length,
            policy.landmark_loss_obj
        )
        if policy.args.debug and num_images_total > 200:
            policy.reporter.info('Break the training due to debugging...')
            break
    
    # report
    policy.reporter.write_epoch_end(
        epoch, cumulative_loss, cumulative_losses, num_batches_each, policy.landmark_loss_obj
    )
