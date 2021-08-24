######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces


def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    sampled_arch = network.module.dync_genotype(True)
    network.module.set_cal_mode('dynamic', sampled_arch)
    #network.module.set_cal_mode( 'urs' )
    network.zero_grad()
    _, logits = network(base_inputs)
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    # update the architecture-weight
    network.module.set_cal_mode( 'joint' )
    network.zero_grad()
    _, logits = network(arch_inputs)
    arch_loss = criterion(logits, arch_targets)
    arch_loss.backward()
    a_optimizer.step()
    # record
    arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
    arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
    arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
    arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
      #print (nn.functional.softmax(network.module.arch_parameters, dim=-1))
      #print (network.module.arch_parameters)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


def get_best_arch(xloader, network, n_samples):
  with torch.no_grad():
    network.eval()
    archs, valid_accs = network.module.return_topK(n_samples), []
    #print ('obtain the top-{:} architectures'.format(n_samples))
    loader_iter = iter(xloader)
    for i, sampled_arch in enumerate(archs):
      network.module.set_cal_mode('dynamic', sampled_arch)
      try:
        inputs, targets = next(loader_iter)
      except:
        loader_iter = iter(xloader)
        inputs, targets = next(loader_iter)

      _, logits = network(inputs)
      val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))

      valid_accs.append( val_top1.item() )
      #print ('--- {:}/{:} : {:} : {:}'.format(i, len(archs), sampled_arch, val_top1))

    best_idx = np.argmax(valid_accs)
    best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
    return best_arch, best_valid_acc


def valid_func(xloader, network, criterion):
  data_time, batch_time = AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  with torch.no_grad():
    network.eval()
    for step, (arch_inputs, arch_targets) in enumerate(xloader):
      arch_targets = arch_targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - end)
      # prediction
      _, logits = network(arch_inputs)
      arch_loss = criterion(logits, arch_targets)
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
      arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
      arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
  return arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  if xargs.dataset == 'cifar10' or xargs.dataset == 'cifar100':
    split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
    cifar_split = load_config(split_Fpath, None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    logger.log('Load split file from {:}'.format(split_Fpath))
  elif xargs.dataset.startswith('ImageNet16'):
    split_Fpath = 'configs/nas-benchmark/{:}-split.txt'.format(xargs.dataset)
    imagenet16_split = load_config(split_Fpath, None, None)
    train_split, valid_split = imagenet16_split.train, imagenet16_split.valid
    logger.log('Load split file from {:}'.format(split_Fpath))
  else:
    raise ValueError('invalid dataset : {:}'.format(xargs.dataset))
  #config_path = 'configs/nas-benchmark/algos/SETN.config'
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  # To split data
  train_data_v2 = deepcopy(train_data)
  train_data_v2.transform = valid_data.transform
  valid_data    = train_data_v2
  search_data   = SearchDataset(xargs.dataset, train_data, train_split, valid_split)
  # data loader
  search_loader = torch.utils.data.DataLoader(search_data, batch_size=config.batch_size, shuffle=True , num_workers=xargs.workers, pin_memory=True)
  valid_loader  = torch.utils.data.DataLoader(valid_data,  batch_size=config.test_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=xargs.workers, pin_memory=True)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  model_config = dict2config({'name': 'SETN', 'C': xargs.channel, 'N': xargs.num_cells,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space'    : search_space}, None)
  logger.log('search space : {:}'.format(search_space))
  search_model = get_cell_based_tiny_net(model_config)
  
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  #logger.log('{:}'.format(search_model))
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {}

  # start training
  start_time, epoch_time, total_epoch = time.time(), AverageMeter(), config.epochs + config.warmup
  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

    search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5 \
                = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, logger)
    logger.log('[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5))
    logger.log('[{:}] search [arch] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, search_a_loss, search_a_top1, search_a_top5))

    genotype, temp_accuracy = get_best_arch(valid_loader, network, xargs.select_num)
    network.module.set_cal_mode('dynamic', genotype)
    valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
    logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}% | {:}'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5, genotype))
    #search_model.set_cal_mode('urs')
    #valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
    #logger.log('[{:}] URS---evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
    #search_model.set_cal_mode('joint')
    #valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
    #logger.log('[{:}] JOINT-evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
    #search_model.set_cal_mode('select')
    #valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
    #logger.log('[{:}] Selec-evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
    # check the best accuracy
    valid_accuracies[epoch] = valid_a_top1

    genotypes[epoch] = genotype
    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict(),
                'genotypes'   : genotypes,
                'valid_accuracies' : valid_accuracies},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)
    with torch.no_grad():
      logger.log('arch-parameters :\n{:}'.format( nn.functional.softmax(search_model.arch_parameters, dim=-1).cpu() ))
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  #logger.log('During searching, the best gentotype is : {:} , with the validation accuracy of {:.3f}%.'.format(genotypes['best'], valid_accuracies['best']))
  genotype, temp_accuracy = get_best_arch(valid_loader, network, xargs.select_num)
  network.module.set_cal_mode('dynamic', genotype)
  valid_a_loss , valid_a_top1 , valid_a_top5 = valid_func(valid_loader, network, criterion)
  logger.log('Last : the gentotype is : {:}, with the validation accuracy of {:.3f}%.'.format(genotype, valid_a_top1))
  # sampling
  """
  with torch.no_grad():
    logger.log('arch-parameters :\n{:}'.format( nn.functional.softmax(search_model.arch_parameters, dim=-1).cpu() ))
  selected_archs = set()
  while len(selected_archs) < xargs.select_num:
    architecture = search_model.dync_genotype()
    selected_archs.add( architecture )
  logger.log('select {:} architectures based on the learned arch-parameters'.format( len(selected_archs) ))

  best_arch, best_acc = None, -1
  state_dict = deepcopy( network.state_dict() )
  for index, arch in enumerate(selected_archs):
    with torch.no_grad():
      search_model.set_cal_mode('dynamic', arch)
      network.load_state_dict( deepcopy(state_dict) )
      valid_a_loss , valid_a_top1 , valid_a_top5  = valid_func(valid_loader, network, criterion)
    logger.log('{:} [{:03d}/{:03d}] : {:125s}, loss={:.3f}, accuracy={:.3f}%'.format(time_string(), index, len(selected_archs), str(arch), valid_a_loss , valid_a_top1))
    if best_arch is None or best_acc < valid_a_top1:
      best_arch, best_acc = arch, valid_a_top1
  logger.log('Find the best one : {:} with accuracy={:.2f}%'.format(best_arch, best_acc))
  """

  logger.log('\n' + '-'*100)
  # check the performance from the architecture dataset
  """
  if xargs.arch_nas_dataset is None or not os.path.isfile(xargs.arch_nas_dataset):
    logger.log('Can not find the architecture dataset : {:}.'.format(xargs.arch_nas_dataset))
  else:
    nasbench101 = TinyNASBenchmarkAPI(xargs.arch_nas_dataset)
    geno      = best_arch
    logger.log('The last model is {:}'.format(geno))
    info = nasbench101.query_by_arch( geno )
    if info is None: logger.log('Did not find this architecture : {:}.'.format(geno))
    else           : logger.log('{:}'.format(info))
    logger.log('-'*100)
  """
  logger.close()
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("SETN")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--select_num',         type=int,   help='The number of selected architectures to evaluate.')
  parser.add_argument('--config_path',        type=str,   help='.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
