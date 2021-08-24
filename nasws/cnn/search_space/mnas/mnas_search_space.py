# Mobile NAS search space

from nasws.cnn.search_space.search_space import CNNSearchSpace, CellSpecTemplate
from collections import namedtuple


MNasOp = namedtuple('MNasOp', ('conv_op', 'kernel_size', 'se_ratio', 'skip_op', 'filter_size', 'layer_num'))


class CellSpec_MNas(CellSpecTemplate):

    pass



class MobileNasSearchSpace(CNNSearchSpace):

    # topology
    # Ops

    def __init__(self):
        pass

