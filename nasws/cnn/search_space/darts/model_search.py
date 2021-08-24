"""
Implement a search space for General NAS model.

Design idea is to make use ModelSpec.

build this from DARTS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# from torch.distributions.categorical import Categorical
import numpy as np
from ..supernet import Supernet
from .genotype import PRIMITIVES
from .dartsbench import DartsModelSpec, Genotype
from .model import NetworkCIFAR, AuxiliaryHeadCIFAR, NetworkImageNet, AuxiliaryHeadImageNet
from .cell import DartsSpaceCellOpEdgeSearch, DartsSpaceCellSearchV2, DartsSpaceCellSearch

class DartsNetworkCIFARSearch(NetworkCIFAR, Supernet):

    def __init__(self, args):

        num_classes = 10
        channel = args.init_channels
        layers = args.layers # total number of layers, equal to num cell
        auxiliary = args.use_auxiliary_in_search
        if args.supernet_cell_type == 'op_on_edge':
            c = DartsSpaceCellOpEdgeSearch
        elif args.supernet_cell_type == 'op_on_edge_fix':
            c = DartsSpaceCellSearchV2
        else:
            c = DartsSpaceCellSearch
        
        Supernet.__init__(self, args) # call this first because it will reset all ops registered to this op
        NetworkCIFAR.__init__(
            self, channel, num_classes, layers, auxiliary,
            None, cell_cls=c, args=args)
        
        self.drop_path_prob = args.path_dropout_rate
        # initialize the arch parameters but may not necessarily use it.

    def forward_oneshot(self, inputs):
        return NetworkCIFAR.forward(self, inputs)
    
    def forward(self, inputs):
        return Supernet.forward(self, inputs)

    def change_genotype(self, genotype):
        self.model_spec_cache = genotype
        for cell in self.cells:
            cell.change_genotype(genotype)

    def convert_to_normal_net(self):
        # To implement and test.
        pass


class DartsSupernetInterface:
    # just an api
    def __init__(self, args) -> None:
        self.args = args
        logging.info('Supernet of Darts search space with DARTS policy enabled')
        self._steps = args.num_intermediate_nodes
        self._multiplier = self._steps
        self.policy = self.args.search_policy
        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        
        num_ops = len(PRIMITIVES)
        self.arch_parameters = [
            1e-3*torch.randn(k, num_ops),
            1e-3*torch.randn(k, num_ops),
            1e-3*torch.randn(k),
            1e-3*torch.randn(k),
        ]
    
    # DARTS policy arch parameters ...
    @property
    def alphas_normal(self):
        return self.arch_parameters[0]
    @property
    def alphas_reduce(self):
        return self.arch_parameters[1]
    
    # For PC-DARTS policy
    @property
    def betas_normal(self):
        return self.arch_parameters[2]
    @property
    def betas_reduce(self):
        return self.arch_parameters[3]

    def _compute_softoneshot_alphas(self, model_spec):
        """Return softoneshot weight for DARTS based model

        Parameters
        ----------
        model_spec : DartsModelSpec 
            (Genotype to be more specific)

        Returns
        -------
        list
            normal_weights, normal_betas, reduce_weights, reduce_betas
        """
        
        num_ops = len(PRIMITIVES)
        res = []

        def _compute_mutation_softmax(w, counter):
            assert w.size() == counter.size()
            if len(w.size()) == 1:
                w = w.view(1, -1)
                counter = counter.view(1, -1)
            n = w.size(1)
            w /= w.sum(dim=1, keepdim=True)

            delta = self.args.softoneshot_alpha
            ind = counter.sum(dim=1, keepdim=True)
            if n == ind[0]:
                w -= ind * delta
                w += counter * n * delta
                w /= w.sum(dim=1, keepdim=True) # normalize again...
            else:
                w -= ind / (n - ind) * delta
                w += counter * n * delta / (n - ind)
            return w

        for alphas, betas, spec in zip(
            [self.alphas_normal, self.alphas_reduce], 
            [self.betas_normal, self.betas_reduce],
            [model_spec.normal, model_spec.reduce]
            ):

            weights = torch.ones_like(alphas)
            # weights /= weights.sum(dim=1, keepdim=True)

            # use a counter variable 
            counter = torch.zeros_like(alphas)
            offset = 0
            for i in range(self._steps):
                for opname, prev_id in spec[i*2: (i+1)*2]:
                    opidx = PRIMITIVES.index(opname)
                    counter[offset+prev_id][opidx] += 1.
                offset += i + 2

            # compute the mapping
            # ind = counter.sum(dim=1, keepdim=True)
            # weights -= ind / (num_ops - ind) * delta
            # weights += counter * num_ops * delta / (num_ops - ind)
            weights = _compute_mutation_softmax(weights, counter)
            res.append(weights)

            # ignore betas for now 
            weights2 = torch.ones_like(betas[0:2])
            counter = torch.zeros_like(betas[0:2])
            # weights2 /= weights2.sum(dim=0, keepdim=True)
            # id1, id2 = spec[0][1], spec[1][1]
            for _, prev_id in spec[:2]:
                counter[prev_id] += 1.
            weights2 = _compute_mutation_softmax(weights2, counter)[0]

            n = 3
            start = 2
            for i in range(self._steps-1):
                end = start + n    
                tw2 = torch.ones_like(betas[start:end])
                counter = torch.zeros_like(betas[start:end])
                for _, prev_id in spec[(i+1) * 2: (i+2) * 2]:
                    counter[prev_id] += 1.
                tw2 = _compute_mutation_softmax(tw2, counter)[0]
                start, n = end, n+1
                weights2 = torch.cat([weights2, tw2], dim=0)
            res.append(weights2)
            
        return res
        
    def _compute_pcdarts_weights(self, reduction):
        if reduction:
            weights = F.softmax(self.alphas_reduce, dim=-1)
            n = 3
            start = 2
            weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
            for i in range(self._steps-1):
                end = start + n
                tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                start = end
                n += 1
                weights2 = torch.cat([weights2, tw2], dim=0)
        else:
            weights = F.softmax(self.alphas_normal, dim=-1)
            n = 3
            start = 2
            weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
            for i in range(self._steps-1):
                end = start + n
                tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                start = end
                n += 1
                weights2 = torch.cat([weights2, tw2], dim=0)
        return weights, weights2

    def genotype_pcdarts(self, method='argmax'):
        
        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :]*W2[j]
                
                # leave it here...
                edges = sorted(range(i + 2), key=lambda x: -max(
                    W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]

                #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
                for j in edges:
                    if method == 'argmax':
                        k_best = None
                        for k in range(len(W[j])):
                            if k != PRIMITIVES.index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                    else:
                        # ignore the none as index 0
                        assert PRIMITIVES.index('none') == 0, 'Redo this logic because none is not at 0 index.'
                        probs = W[j][1:]
                        probs /= probs.sum()
                        k_best = 1 + np.random.choice(np.arange(probs.shape[0]), p=probs)
                    
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(F.softmax(
            self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        gene_reduce = _parse(F.softmax(
            self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())

        concat = range(2+self._steps-self._steps, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concatc
        )
        return DartsModelSpec.from_darts_genotype(genotype)

    def genotype_darts(self, method='argmax'):    
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(
                    W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    if method == 'argmax':
                        k_best = None
                        for k in range(len(W[j])):
                            # manual filter none here...
                            if k != PRIMITIVES.index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                    else:
                        # ignore the none as index 0
                        assert PRIMITIVES.index('none') == 0, 'Redo this logic because none is not at 0 index.'
                        probs = W[j][1:]
                        probs /= probs.sum()
                        k_best = 1 + np.random.choice(np.arange(probs.shape[0]), p=probs)
                    
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(
            F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(
            F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return DartsModelSpec.from_darts_genotype(genotype)
    

    def genotype(self, method='argmax'):
        if 'pcdarts' in self.policy:
            return self.genotype_pcdarts(method)         
        else:
            return self.genotype_darts(method)

    def new(self, parallel=None):
        import copy
        return copy.deepcopy(self)

    def forward_softoneshot_cells(self, s0, s1):
        logits_aux = None
        normal_op, normal_node, reduce_op, reduce_node = self._compute_softoneshot_alphas(self.model_spec_cache)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = reduce_op
                weights2 = reduce_node
            else:
                weights = normal_op
                weights2 = normal_node
            if self.mode == 'pcdarts':
                s0, s1 = s1, cell.forward_pcdarts(s0, s1, weights, weights2)
            else:
                s0, s1 = s1, cell.forward_darts(s0, s1, weights)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        
        return s1, logits_aux

    def forward_gdas_cells(self, s0, s1):
        def get_gumbel_prob(xins):
            while True:
                gumbels = -torch.empty_like(xins).exponential_().log()
                logits  = (xins.log_softmax(dim=1) + gumbels) / self.args.gdas_tau
                probs   = nn.functional.softmax(logits, dim=1)
                index   = probs.max(-1, keepdim=True)[1]
                one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                    continue
                else: break
            return hardwts, index
        
        normal_hardwts, normal_index = get_gumbel_prob(self.alphas_normal)
        reduce_hardwts, reduce_index = get_gumbel_prob(self.alphas_reduce)

        logits_aux = None
        for i, cell in enumerate(self.cells):
            if cell.reduction: 
                hardwts, index = reduce_hardwts, reduce_index
            else: 
                hardwts, index = normal_hardwts, normal_index
            s0, s1 = s1, cell.forward_gdas(s0, s1, hardwts, index)
            # generate the auxiliary head.
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        return s1, logits_aux

    def forward_pcdarts_cells(self, s0, s1):
        logits_aux = None
        for i, cell in enumerate(self.cells):
            weights, weights2 = self._compute_pcdarts_weights(cell.reduction)
            s0, s1 = s1, cell.forward_pcdarts(s0, s1, weights, weights2)
            # generate the auxiliary head.
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        return s1, logits_aux

    def forward_darts_cells(self, s0, s1):
        logits_aux = None
        for i, cell in enumerate(self.cells):
            weights, weights2 = self._compute_pcdarts_weights(cell.reduction)
            try:
                s0, s1 = s1, cell.forward_darts(s0, s1, weights)
            except Exception as e:
                raise e
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        return s1, logits_aux
    


class DartsNetworkCIFARSearchDARTS(DartsNetworkCIFARSearch, DartsSupernetInterface):

    def __init__(self, args):
        DartsNetworkCIFARSearch.__init__(self, args)
        DartsSupernetInterface.__init__(self, args)
        self.mode = self.args.search_policy

    def forward_softoneshot(self, inputs):
        # op - alphas, node - beta
        s0 = s1 = self.stem(inputs)
        s1, logits_aux = self.forward_softoneshot_cells(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
    
    def forward_darts(self, inputs):
        s0 = s1 = self.stem(inputs)
        logits_aux = None
        s1, logits_aux = self.forward_darts_cells(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

    def forward_pcdarts(self, inputs):
        s0 = s1 = self.stem(inputs)
        logits_aux = None
        s1, logits_aux = self.forward_pcdarts_cells(s0, s1)        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux 
    
    def forward_gdas(self, inputs):
        s0 = s1 = self.stem(inputs)
        s1, logits_aux = self.forward_gdas_cells(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux 