import torch
import torch.nn as nn
import torch.nn.functional as F
from nasws.rnn.darts_policy.genotypes import PRIMITIVES, Genotype
from torch.autograd import Variable
from nasws.rnn.darts_policy.darts_model import DARTSCell, RNNModel


class DARTSCellSearch(DARTSCell):

    def __init__(self, ninp, nhid, dropouth, dropoutx, num_intermediate_nodes, handle_hidden_mode=None):
        super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None,
                                              num_intermediate_nodes=num_intermediate_nodes,
                                              handle_hidden_mode=handle_hidden_mode)
        self.bn = nn.BatchNorm1d(nhid, affine=False)
        self.handle_hidde_mode = handle_hidden_mode

    def cell(self, x, h_prev, x_mask, h_mask):
        """
        overriding the cell method of the DARTSCell class.
        During arch search, the forward of the cell is different,
        the node output is computed using the MIXED operation on
        all possible previous nodes (index with lower value than
        the current node). See paper page 2 for the formula to compute
        the node (i), replace the normal operation in that formula
        with the definition of the MIXED op at page 3 to understand
        the computation of the node (i) output during arch search.
        :param x: input
        :param h_prev: first hidden state
        :param x_mask: input dropout mask
        :param h_mask: hidden dropout mask
        """

        # preparing the first state usign method of parent class
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        s0 = self.bn(s0)

        # apply softmax to obtain probability distribution
        # for operations on each edges
        # self.weights are assigned in the RNNModelSearch class, line 94
        probs = F.softmax(self.weights, dim=-1)

        offset = 0
        states = s0.unsqueeze(0)
        for i in range(self.num_intermediate_nodes):
            if self.training:
                masked_states = states * h_mask.unsqueeze(0)
            else:
                masked_states = states

            # each node is based on all its predecessors, here we compute the
            # product of all the previous node outputs for the matrix Ws[i] with
            # the params (FC layer) for the node 'i'. Example: intermediate node 2
            # can have as previous nodes n0 and n1. The states n0 and n1 are
            # multiplied for the matrix Ws[2] (FC layer towards node 2)
            ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i + 1, -1, 2 * self.nhid)
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            # here we compute the last part to obtain the node 'i' as weighted sum
            # of the MIXED operation (see paper) applied on all the possible previous edges
            s = torch.zeros_like(s0)

            # for each possible operation:
            for k, name in enumerate(PRIMITIVES):
                if name == 'none':
                    continue
                fn = self._get_activation(name)

                # each function is applied on the hidden
                # states of all the possible previous nodes
                unweighted = states + c * (fn(h) - states)

                # weighted sum of the operation k applied to
                # the previous nodes by using the prob of operation k
                # stored in the vectors describing the edges between
                # node i and all the possible previous nodes (offset : offset + i + 1)
                s += torch.sum(probs[offset:offset + i + 1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)

            s = self.bn(s)
            states = torch.cat([states, s.unsqueeze(0)], 0)
            offset += i + 1

        # output as the average of the output of the intermediate nodes
        output = torch.mean(states[-self.num_intermediate_nodes:], dim=0)

        if self.handle_hidde_mode == 'ACTIVATION':
            output = F.tanh(output)

        return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None)
        self._args = args
        self._initialize_arch_parameters()

    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
        """
        initialization of the alpha parameters
        that describes the architecture representation
        """

        # k is the number of edges for the longest path
        # possible in a DAG having n nodes. n(n-1)/2
        # that is the sum of the first n natural numbers
        k = sum(i for i in range(1, self.num_intermediate_nodes + 1))

        # returns a torch tensor of dimension k x |operations| initialized randomly
        weights_data = torch.randn(k, len(PRIMITIVES)).mul_(1e-3)

        # making the arch weights a variable with gradient accumulator enabled
        self.weights = Variable(weights_data.cuda(), requires_grad=True)

        self._arch_parameters = [self.weights]

        for rnn in self.rnns:
            rnn.weights = self.weights

    def arch_parameters(self):
        return self._arch_parameters

    def _loss(self, hidden, input, target):
        log_prob, hidden_next = self(input, hidden, return_h=False)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
        return loss, hidden_next

    def genotype(self):

        def _parse(probs):
            """
            build the discrete representation of the cell
            :param probs: tensor of dim (|max_edges| x |operations| representing the prob distribution of the ops
            :return:
            """
            gene = []
            start = 0

            # 'i' is the index regarding the edge to the ith intermediate node
            for i in range(self.num_intermediate_nodes):
                end = start + i + 1

                # selecting the alpha vectors dedicated to the incoming edges of intermediate node i
                # for i = 2, get the vectors regarding edges: e(0,2), e(1,2)
                # for i = 3, get the vectors regarding edges: e(0,3), e(1,3), e(2,3)
                W = probs[start:end].copy()

                # among the vectors of the valid edges, select the vector of
                # the edge with the highest probable operation, this is for
                # the constraint that each node has only 1 incoming edge (see paper)
                j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]

                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k

                # appending tuple describing the edge towards the ith node,
                # describing the activation function and the previous node (j)
                gene.append((PRIMITIVES[k_best], j))

                start = end

            return gene

        gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
        genotype = Genotype(recurrent=gene, concat=range(self.num_intermediate_nodes + 1)[-self.num_intermediate_nodes:])
        return genotype
