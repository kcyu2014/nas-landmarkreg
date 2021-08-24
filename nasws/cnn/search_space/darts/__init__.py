"""
Darts base search space

Support both CIFAR and Imagenet space

"""
from .dartsbench import DARTSBench, DartsModelSpec, Genotype
from .darts_search_space import DartsSearchSpace
from .model import NetworkCIFAR, NetworkImageNet
from .model_search import DartsNetworkCIFARSearch
from .model_search_imagenet import DartsNetworkImageNetSearch