import logging

from ..supernet import Supernet
from .cell import DartsSpaceCellSearchV2, DartsSpaceCellSearch, DartsSpaceCellOpEdgeSearch
from .model_search import DartsSupernetInterface
from .model import NetworkImageNet


class DartsNetworkImageNetSearch(NetworkImageNet, Supernet):

    def __init__(self, args=None):
        num_classes = 1000
        channel = args.init_channels
        layers = args.layers # total number of layers, equal to num cell
        auxiliary = args.use_auxiliary_in_search
        if args.supernet_cell_type == 'op_on_edge':
            c = DartsSpaceCellOpEdgeSearch
        elif args.supernet_cell_type == 'op_on_edge_fix':
            c = DartsSpaceCellSearchV2
        else:
            c = DartsSpaceCellSearch
        logging.info(f'DartsNetworkImagenetSearch with cell type {c}')

        Supernet.__init__(self, args)
        NetworkImageNet.__init__(self, channel, num_classes, layers, auxiliary,
                                                        None, cell_cls=c,
                                                        args=args)
        self.drop_path_prob = args.path_dropout_rate

    def change_genotype(self, genotype):
        for cell in self.cells:
            cell.change_genotype(genotype)

    def convert_to_normal_net(self):
        # To implement and test.
        pass
    
    def forward(self, inputs):
        # override this to avoid any confusion
        return Supernet.forward(self, inputs)

    def forward_oneshot(self, inputs):
        return NetworkImageNet.forward(self, inputs)


class DartsNetworkImageNetSearchDARTS(DartsNetworkImageNetSearch, DartsSupernetInterface):
    
    def __init__(self, args) -> None:
        DartsNetworkImageNetSearch.__init__(self, args)
        # as DARTS supernet interface has the same functions...
        DartsSupernetInterface.__init__(self, args)
        self.mode = self.args.search_policy
    
    def forward_pcdarts(self, inputs):
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        s1, logits_aux = self.forward_pcdarts_cells(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
    

    def forward_softoneshot(self, inputs):
        # op - alphas, node - beta
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        s1, logits_aux = self.forward_softoneshot_cells(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
    
    def forward_darts(self, inputs):
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        logits_aux = None
        s1, logits_aux = self.forward_darts_cells(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
    
    def forward_gdas(self, inputs):
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        s1, logits_aux = self.forward_gdas_cells(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux 