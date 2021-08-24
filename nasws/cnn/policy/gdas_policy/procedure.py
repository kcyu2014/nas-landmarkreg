# import nasws
# import time
# import logging
# import torch.nn as nn
# from nasws.cnn.procedures.train_search_procedure import _summarize_shared_train


# def adjust_tau_value(epoch, args):
#     tau = args.tau_max - (args.tau_max-args.tau_min) * epoch / (args.epochs-1) 
#     args.tmp_tau = tau
#     return tau


# def gdas_train_model(train_queue, valid_queue, model, criterion, optimizer, lr, args, architect, sampler=None):
#     objs = nasws.cnn.utils.AverageMeter()
#     top1 = nasws.cnn.utils.AverageMeter()
#     top5 = nasws.cnn.utils.AverageMeter()

#     arch_objs = nasws.cnn.utils.AverageMeter()
#     arch_top1 = nasws.cnn.utils.AverageMeter()
#     arch_top5 = nasws.cnn.utils.AverageMeter()

#     # track the timer
#     timers = {k: nasws.cnn.utils.AverageMeter() for k in ['batch', 'data', 'sampler', 'model']}

#     end = time.time()

#     model.train()
#     tau = adjust_tau_value(args.current_epoch, args)
#     logging.info(f'--> GDAS Tau {tau}')

#     for step, (input, target) in enumerate(train_queue):
#         bstart = end
#         n = input.size(0)
#         input = input.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)

#         timers['data'].update(time.time() - end)
        
#         # sampler
#         if args.debug and step > 10:
#             logging.warning('Testing only. Break after 10 batches.')
#             break

#         end = time.time()
#         if sampler:
#             model = sampler(model, None, args)
#             # print("Activate sampler")
#         timers['sampler'].update(time.time() - end)
        
#         # this is a fairly simple step function logic. update the architecture in each step, before updating the
#         # weight itself.
#         end = time.time()
#         optimizer.zero_grad()
#         logits, _ = model(input)
#         loss = criterion(logits, target)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#         optimizer.step()
#         timers['model'].update(time.time() - end)

#         prec1, prec5 = nasws.cnn.utils.accuracy(logits, target, topk=(1, 5))
#         objs.update(loss.item(), n)
#         top1.update(prec1.item(), n)
#         top5.update(prec5.item(), n)

#         # get a random minibatch from the search queue with replacement
#         arch_inputs, arch_targets = next(iter(valid_queue))
#         arch_inputs = arch_inputs.cuda()
#         arch_targets = arch_targets.cuda(non_blocking=True)
#         architect.step()
#         # record
#         arch_prec1, arch_prec5 = nasws.cnn.utils.accuracy(logits.data, arch_targets.data, topk=(1, 5))
#         arch_objs.update(arch_loss.item(),  arch_inputs.size(0))
#         arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
#         arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

#         timers['batch'].update(time.time() - bstart)

#         if step % 50 == 0:
#         # if step % args.report_freq == 0:
#             _summarize_shared_train(step, objs.avg, objs.avg, top1.avg, top5.avg, lr, prefix='Base')
#             _summarize_shared_train(step, arch_objs.avg, arch_objs.avg, arch_top1.avg, arch_top5.avg, lr, prefix='Arch')
        
#         end = time.time()

#     return top1.avg, objs.avg
