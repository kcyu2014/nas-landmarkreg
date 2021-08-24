import argparse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--epochs', type=int, default=35, help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=15, help='num of training epochs')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tau_min', type=float, default=.1, help='The minimum tau for Gumbel')
parser.add_argument('--tau_max', type=float, default=10., help='The maximum tau for Gumbel')
parser.add_argument('--unrolled', default=False, action='store_true', help='The maximum tau for Gumbel')

gdas_args = parser.parse_args('')
