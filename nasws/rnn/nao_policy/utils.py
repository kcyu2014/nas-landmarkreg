import torch
import torch.nn as nn
import os, shutil
import numpy as np
import math
from torch.autograd import Variable
from collections import namedtuple

STEPS = 11

PRIMITIVES = [
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]

"""
<sos>     0
0         1
1         2
2         3
3         4
4         5
5         6
6         7
7         8
8         9
9         10
10        11
tanh      12
relu      13
sigmoid   14
identity  15
"""
Genotype = namedtuple('Genotype', 'recurrent concat')

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, cuda):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    if cuda:
        data = data.cuda()
    return data


def controller_batchify(data, bsz, cuda):
    data_size = data.size()
    nbatch = data_size[0] // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(nbatch, bsz, *data_size[1:]).contiguous()
    # print(data.size())
    if cuda:
        data = data.cuda()
    return data


def controller_get_batch(data, i, evaluation=False):
    data = Variable(data[i], volatile=evaluation)
    return data


def get_batch(source, i, bptt, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else bptt, len(source) - 1 - i)
    data = Variable(source[i:i + seq_len], volatile=evaluation)
    target = Variable(source[i + 1:i + 1 + seq_len])
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, epoch, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
    torch.save({'epoch': epoch + 1}, os.path.join(path, 'misc.pt'))


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = embed._backend.Embedding.apply(words, masked_embed_weight,
                                       padding_idx, embed.max_norm, embed.norm_type,
                                       embed.scale_grad_by_freq, embed.sparse
                                       )
    return X


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


def mask2d(B, D, keep_prob, cuda=True):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m = Variable(m, requires_grad=False)
    if cuda:
        m = m.cuda()
    return m


def generate_arch(n, num_nodes, num_ops):
    def _get_arch():
        arch = []
        for i in range(1, num_nodes + 1):
            pred = np.random.randint(0, i)
            act = np.random.randint(0, num_ops)
            arch.extend([pred, act])
        return arch

    archs = [_get_arch() for i in range(n)]  # [[arch]]
    return archs


def build_arch(arch):
    if arch is None:
        return None
    # assume arch is the format [idex, op ...] where index is in [0, 10] and op in [0, 3]
    arch = list(map(int, arch.strip().split()))
    return arch


def parse_arch_to_seq(cell, num_nodes):
    seq = []

    for i in range(num_nodes):
        pred = cell[2 * i] + 1
        act = cell[2 * i + 1] + num_nodes + 1
        seq.extend([pred, act])
    return seq


def parse_seq_to_arch(seq, num_nodes):
    n = len(seq)
    arch = []
    for i in range(num_nodes):
        pred = seq[2 * i] - 1
        act = seq[2 * i + 1] - num_nodes - 1
        arch.extend([pred, act])
    return arch


def genotype_id_from_geno(genotype):

    previous_slot = 0
    num_operations = len(PRIMITIVES)

    for index, gene in enumerate(genotype.recurrent):
        operation = gene[0]

        possible_relative_slots = (index + 1) * num_operations

        operation_id = PRIMITIVES.index(operation)

        prev_node_id = gene[1]

        relative_slot = num_operations * prev_node_id + operation_id

        previous_slot = relative_slot + (previous_slot * possible_relative_slots)

    return previous_slot


def pairwise_accuracy(la, lb):
    N = len(la)
    assert N == len(lb)
    total = 0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if la[i] > la[j] and lb[i] > lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            if la[i] == la[j] and lb[i] == lb[j]:
                count += 1
            total += 1
    return float(count) / total


def hamming_distance(la, lb):
    N = len(la)
    assert N == len(lb)

    def _hamming_distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c

    dis = 0
    for i in range(N):
        line1 = la[i]
        line2 = lb[i]
        dis += _hamming_distance(line1, line2)
    return dis / N


def normalize_target(target_list):
    min_val = min(target_list)
    max_val = max(target_list)
    res = [(i - min_val) / (max_val - min_val) for i in target_list]
    return res


def parse_arch(arch):
    return list(map(int, arch.strip().split()))

def get_genotype(arch, num_nodes, operations_list, use_avg_leaf=False):
    gene = []

    concat = set([elem for elem in range(1, num_nodes + 1)])

    for node in range(num_nodes):
        previous_node_id = arch[2 * node]
        activation_id = arch[2*node + 1]
        activation_name = operations_list[activation_id]
        gene.append((activation_name, previous_node_id))
        if previous_node_id in concat:
            concat.remove(previous_node_id)

    if use_avg_leaf:
        concat = sorted(list(concat))
    else:
        concat = range(num_nodes + 1)[-num_nodes:]

    # print('UTILS METHOD OUTPUT NODES USED: {}'.format(concat))
    return Genotype(recurrent=gene, concat=concat)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


