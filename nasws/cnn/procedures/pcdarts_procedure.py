import logging
import torch.nn as nn
import nasws
from .train_search_procedure import _summarize_shared_train


def pcdarts_train_procedure(
    train_queue, valid_queue, model, criterion, optimizer, lr, args, architect=None, sampler=None):
    objs = nasws.cnn.utils.AverageMeter()
    arch_objs = nasws.cnn.utils.AverageMeter()
    top1 = nasws.cnn.utils.AverageMeter()
    top5 = nasws.cnn.utils.AverageMeter()
    logging_prefix = ''

    for step, (input, target) in enumerate(train_queue):
        if args.debug:
            if step > 10:
                logging.debug('Debugging here, break after 100 iter')
                break
        
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        if sampler:
            model = sampler(model, architect, args)
            # print("Activate sampler")

        # get a random minibatch from the search queue with replacement
        #input_search, target_search = next(iter(valid_queue))
        
        if args.current_epoch >= args.epochs and architect:
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            arch_loss = architect.step(
                input, target, input_search, target_search, lr, optimizer, unrolled=args.policy_args.unrolled
            )
            logging_prefix = 'arch train'
            arch_objs.update(arch_loss, input_search.size(0))
        else:
            logging_prefix = 'arch warmup'
        
        optimizer.zero_grad()
        logits, _ = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = nasws.cnn.utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        if step % args.report_freq == 0:
            _summarize_shared_train(step, objs.avg, arch_objs.avg, top1.avg, top5.avg, lr, prefix=logging_prefix)
        
    return top1.avg, objs.avg
