##################################################
# modified from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
##################################################
import os, sys, time, glob, random, argparse
import numpy as np, collections
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Categorical
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, SearchDataset
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from aa_nas_api   import AANASBenchAPI
from models       import CellStructure, get_search_spaces
from R_EA import train_and_eval


class Policy(nn.Module):

  def __init__(self, max_nodes, search_space):
    super(Policy, self).__init__()
    self.max_nodes    = max_nodes
    self.search_space = deepcopy(search_space)
    self.edge2index   = {}
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        self.edge2index[ node_str ] = len(self.edge2index)
    self.arch_parameters = nn.Parameter( 1e-3*torch.randn(len(self.edge2index), len(search_space)) )

  def generate_arch(self, actions):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        op_name  = self.search_space[ actions[ self.edge2index[ node_str ] ] ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )

  def genotype(self):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = self.arch_parameters[ self.edge2index[node_str] ]
          op_name = self.search_space[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return CellStructure( genotypes )
    
  def forward(self):
    alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
    return alphas


class ExponentialMovingAverage(object):
  """Class that maintains an exponential moving average."""

  def __init__(self, momentum):
    self._numerator   = 0
    self._denominator = 0
    self._momentum    = momentum

  def update(self, value):
    self._numerator = self._momentum * self._numerator + (1 - self._momentum) * value
    self._denominator = self._momentum * self._denominator + (1 - self._momentum)

  def value(self):
    """Return the current value of the moving average"""
    return self._numerator / self._denominator


def select_action(policy):
  probs = policy()
  m = Categorical(probs)
  action = m.sample()
  #policy.saved_log_probs.append(m.log_prob(action))
  return m.log_prob(action), action.cpu().tolist()


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  assert xargs.dataset == 'cifar10', 'currently only support CIFAR-10'
  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
  cifar_split = load_config(split_Fpath, None, None)
  train_split, valid_split = cifar_split.train, cifar_split.valid
  logger.log('Load split file from {:}'.format(split_Fpath))
  config_path = 'configs/nas-benchmark/algos/R-EA.config'
  config = load_config(config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  # To split data
  train_data_v2 = deepcopy(train_data)
  train_data_v2.transform = valid_data.transform
  valid_data    = train_data_v2
  search_data   = SearchDataset(xargs.dataset, train_data, train_split, valid_split)
  # data loader
  train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split) , num_workers=xargs.workers, pin_memory=True)
  valid_loader  = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split), num_workers=xargs.workers, pin_memory=True)
  logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))
  extra_info = {'config': config, 'train_loader': train_loader, 'valid_loader': valid_loader}
  
  search_space = get_search_spaces('cell', xargs.search_space_name)
  policy    = Policy(xargs.max_nodes, search_space)
  optimizer = torch.optim.Adam(policy.parameters(), lr=xargs.learning_rate)
  eps       = np.finfo(np.float32).eps.item()
  baseline  = ExponentialMovingAverage(xargs.EMA_momentum)
  logger.log('policy    : {:}'.format(policy))
  logger.log('optimizer : {:}'.format(optimizer))
  logger.log('eps       : {:}'.format(eps))

  # nas dataset load
  if xargs.arch_nas_dataset is None or not os.path.isfile(xargs.arch_nas_dataset):
    logger.log('Can not find the architecture dataset : {:}.'.format(xargs.arch_nas_dataset))
    nasbench101 = None
  else:
    logger.log('{:} build NAS-Benchmark-API from {:}'.format(time_string(), xargs.arch_nas_dataset))
    nasbench101 = AANASBenchAPI(xargs.arch_nas_dataset)
  logger.log('{:} use nasbench101 : {:}'.format(time_string(), nasbench101))

  # REINFORCE
  # attempts = 0
  for istep in range(xargs.RL_steps):
    log_prob, action = select_action( policy )
    arch   = policy.generate_arch( action )
    reward = train_and_eval(arch, nasbench101, extra_info)

    baseline.update(reward)
    # calculate loss
    policy_loss = ( -log_prob * (reward - baseline.value()) ).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    logger.log('step [{:3d}/{:3d}] : average-reward={:.3f} : policy_loss={:.4f} : {:}'.format(istep, xargs.RL_steps, baseline.value(), policy_loss.item(), policy.genotype()))
    #logger.log('----> {:}'.format(policy.arch_parameters))
    logger.log('')

  best_arch = policy.genotype()

  if nasbench101 is not None:
    info = nasbench101.query_by_arch( best_arch )
    if info is None: logger.log('Did not find this architecture : {:}.'.format(best_arch))
    else           : logger.log('{:}'.format(info))
  logger.log('-'*100)

  logger.close()
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Regularized Evolution Algorithm")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--learning_rate',      type=float, help='The learning rate for REINFORCE.')
  parser.add_argument('--RL_steps',           type=int,   help='The steps for REINFORCE.')
  parser.add_argument('--EMA_momentum',       type=float, help='The momentum value for EMA.')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
