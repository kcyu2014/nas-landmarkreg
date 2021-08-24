"""
Implementation of ProxylessNas search space.

"""
from ..darts.darts_search_space import DartsSearchSpace

from .model import ProxylessNASNets
from .proxylessbench import ProxylessNasBench


class ProxylessNasSearchSpace(DartsSearchSpace):

    def __init__(self, args):
        super(DartsSearchSpace, self).__init__(args)
        self.topology_fn = ProxylessNASNets
        self.dataset = ProxylessNasBench(args.data + '/proxylessnas/proxylessnasbench_v1.json')
        self._construct_search_space()

    def random_topology(self):
        pass
