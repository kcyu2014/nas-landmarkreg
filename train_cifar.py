import os
from datetime import timedelta

import shutil
import sys
import time
import glob
import numpy as np
import torch

import nasws.cnn.utils
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from nasws.cnn.search_space.nasbench101.nasbench_api_v2 import NASBench_v2
from nasws.cnn.search_space.nasbench101.util import display_cell
from nasws.cnn.search_space.nasbench101.model import NasBenchNet
from nasws.cnn.search_space.nasbench101.genotype import model_spec_demo, test_model_max
from nasws.cnn.policy.darts_policy.model import NetworkCIFAR
import nasws.cnn.policy.darts_policy.genotypes as cifar_genotype

# depending on this
from nasws.cnn.search_space.nasbench101.lib import config as _config
from nasws.rnn.search_configs import str2bool

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--policy', type=str, default='NASBench', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--n_workers', type=int, default=8, help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=32, help='eval batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=128, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--num_cells', type=int, default=3, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='random', help='which architecture to use, should be hash number here.')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save_every_epochs', type=int, default=40, help='save the architecture every ? epochs')
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['sgd', 'rmsprop'], help="Define the optimizer, default rmsprop")
parser.add_argument('--run_comment', type=str, default='runid-0', help='This is to helping identifying different run.')
parser.add_argument('--resume', type=str2bool, default='True')
parser.add_argument('--debug', action='store_true', default=False)

args = parser.parse_args()

# args.save = 'experiments/{}-eval-{}'.format(args.save, args.run_comment)
try:
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
except FileExistsError as e:
  logging.info("File exists, try to reload later.")


logger = utils.get_logger(
  'CIFAR 10 Reproduce.',
  file_handler=utils.get_file_handler(os.path.join(args.save, 'log.txt'))
)

# logger.info('Testing from logger.')
# logging.info('Testing from logging.')
# DEBUG
torch.autograd.set_detect_anomaly(True)


def network_fn(args):
  genotype = arch_to_genotype(args.arch, args.policy)
  if args.policy.lower() == 'nasbench':
    nasbench_config = _config.build_config()
    nasbench_config["stem_filter_size"] = args.init_channels
    nasbench_config["num_stacks"] = args.layers
    print(f'NASBench 101 model with {args.layers} numStacks and {args.init_channels} channels.')
    model = NasBenchNet(3, genotype, nasbench_config)
    args.optimizer = 'sgd'
    # args.optimizer = 'rmsprop'
  elif args.policy.lower() in ['nao', 'darts', 'enas', 'random', 'spos', 'rankloss']:
    CIFAR_CLASSES = 10
    model = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype, args)
    if torch.cuda.device_count() > 1:
      logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
    model = nn.DataParallel(model)
    args.optimizer = 'sgd'
  else:
    raise NotImplementedError
  return model, genotype


def arch_to_genotype(arch, method):
  if method.lower() == 'nasbench':
    if arch == 'random':
      # genotype = random_spec(nasbench=NASBench_v2('/home/yukaiche/pycharm/nasbench/nasbench_only108.tfrecord'))
      genotype = model_spec_demo
    elif arch == 'test-1':
      genotype = test_model_max(1)
    else:
      # Read as 'hash'
      logger.info("Treat the arch as hash: {}".format(arch))
      nasbench_v2 = NASBench_v2('./data/nasbench/nasbench_only108.tfrecord', only_hash=False)
      logger.info("Load from NASBench dataset v2")
      display_cell(nasbench_v2.query_hash(arch))
      model_arch = nasbench_v2.hash_to_model_spec(arch)
      logger.info('Is this valid? {}'.format(nasbench_v2.is_valid(model_arch)))
      return model_arch
  elif method.lower() == 'nao':
    genotype = cifar_genotype.nao_final_genotypes_by_seed(int(arch))
    logging.info("Using NAO final searched model under seed {}: Genotype {}".format(arch, genotype))
  elif method.lower() == 'darts':
    genotype = cifar_genotype.darts_final_genotype_by_seed(int(arch))
    logging.info(("Using DARTS search: {}".format(genotype)))
  elif method.lower() == 'enas':
    genotype = cifar_genotype.enas_final_genotypes_by_seed(int(arch))
    logging.info("Using ENAS final search model under seed {}: Genotype {}".format(arch, genotype))
  elif method.lower() == 'random':
    genotype = cifar_genotype.random_generate_genotype_by_seed(int(arch))
    logging.info("Using guided random final search model under seed {}: Genotype {}".format(arch, genotype))
  elif method.lower() == 'spos':
    genotype = cifar_genotype.spos_generate_genotype_by_seed(int(arch))
    logging.info("Using SPOS final search model under seed {}: Genotype {}".format(arch, genotype))
  elif method.lower() == 'rankloss':
    genotype = cifar_genotype.rankloss_generate_genotype_by_seed(int(arch))
    offset = 0
    genotype = genotype[offset]
    logging.info("Using rankloss final search model under seed {}: Genotype {}".format(arch, genotype))

  else:
    raise NotImplementedError
  return genotype


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # Get genotype and network function.
  model, genotype = network_fn(args)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  logging.info("Using optimizer {}".format(args.optimizer))
  if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
    )
  elif args.optimizer == 'rmsprop':
    optimizer = torch.optim.RMSprop(
      model.parameters(),
      args.learning_rate,
      eps=1.0,
      weight_decay=args.weight_decay,
      momentum=args.momentum,
      # NOTE(kyu) : alpha is different in TF?
    )
  else:
    raise ValueError(f"Optimizer not support here! {args.optimizer}")

  train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_length)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.n_workers)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=args.n_workers)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  init_epoch = 0

  # Before training, load
  if args.resume:
    model_save_path = os.path.join(args.save, 'checkpoint.pt')
    if os.path.exists(model_save_path):
      state = torch.load(model_save_path)
      init_epoch = state['epoch']
      model.load_state_dict(state['model_state'])
      optimizer.load_state_dict(state['optimizer_state'])
      scheduler.load_state_dict(state['scheduler_state'])
      logging.info(f"Resuming from previous checkpoint. At epoch {init_epoch + 1}, best accuracy {state['best_acc']}")

  for epoch in range(init_epoch, args.epochs):
    epoch_start_time = time.time()
    epoch = epoch
    logger.info('\n EPOCH {} STARTED.'.format(epoch))

    # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    scheduler.step()

    # logs info
    epoch_end_time = time.time()
    logger.info('-' * 89)
    logger.info('| end of epoch {:3d} | time: {:5.2f}s '
                '| lr {:5.2f}'
                '| train loss {:5.2f} | train acc {:8.2f}'
                '| valid loss {:5.2f} | valid acc {:8.2f}'.format(
      epoch, (epoch_end_time - epoch_start_time),
      scheduler.get_lr()[0],
      train_obj, train_acc, valid_obj, valid_acc))
    logger.info('-' * 89)

    # For saving the entire training.
    state = {
      'epoch': epoch,
      'model_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'scheduler_state': scheduler.state_dict(),
      'best_acc': valid_acc
    }
    
    torch.save(state, os.path.join(args.save, 'checkpoint.pt'))
    if epoch > 0 and (epoch + 1) % args.save_every_epochs == 0:
      print("")
      shutil.copyfile(os.path.join(args.save, 'checkpoint.pt'), os.path.join(args.save, 'checkpoint-{}.pt'.format(epoch + 1)))
    est_time = (epoch_end_time - epoch_start_time) * (args.epochs - epoch)
    logging.info('Time per epoch: %s, Est. complete in: %s' % (
      str(timedelta(seconds=epoch_end_time - epoch_start_time)),
      str(timedelta(seconds=est_time))))

def train(train_queue, model, criterion, optimizer):
  """ Train one epoch """
  objs = nasws.cnn.utils.AverageMeter()
  top1 = nasws.cnn.utils.AverageMeter()
  top5 = nasws.cnn.utils.AverageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    # input = Variable(input)
    # target = Variable(target)
    input = input.cuda()
    target = target.cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = nasws.cnn.utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(utils.to_item(loss), n)
    top1.update(utils.to_item(prec1), n)
    top5.update(utils.to_item(prec5), n)

    if step % args.report_freq == 0:
      logging.info('train step: %03d obj: %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = nasws.cnn.utils.AverageMeter()
  top1 = nasws.cnn.utils.AverageMeter()
  top5 = nasws.cnn.utils.AverageMeter()
  with torch.no_grad():
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda()

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = nasws.cnn.utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(utils.to_item(loss), n)
      top1.update(utils.to_item(prec1), n)
      top5.update(utils.to_item(prec5), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

