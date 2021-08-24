from __future__ import print_function

from collections import defaultdict

from datetime import datetime
import os
import json

import numpy as np
from torch.utils.data import DataLoader

import logging


try:
    import pygraphviz as pgv
    enable_graph = True
except:
    print("Cannot import graphviz package")
    enable_graph = False

import torch
from torch.autograd import Variable


##########################
# Network visualization
##########################

def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('x'):
        color = 'white'
    elif label.startswith('h'):
        color = 'skyblue'
    elif label == 'tanh':
        color = 'yellow'
    elif label == 'ReLU':
        color = 'pink'
    elif label == 'identity':
        color = 'orange'
    elif label == 'sigmoid':
        color = 'greenyellow'
    elif label == 'avg':
        color = 'seagreen3'
    else:
        color = 'white'

    if not any(label.startswith(word) for word in  ['x', 'avg', 'h']):
        label = f"{label}\n({node_id})"

    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style,
    )

def draw_network(dag, path):
    makedirs(os.path.dirname(path))
    if enable_graph:
        graph = pgv.AGraph(directed=True, strict=True,
                           fontname='Helvetica', arrowtype='open') # not work?

        checked_ids = [-2, -1, 0]

        if -1 in dag:
            add_node(graph, -1, 'x[t]')
        if -2 in dag:
            add_node(graph, -2, 'h[t-1]')

        add_node(graph, 0, dag[-1][0].name)

        for idx in dag:
            for node in dag[idx]:
                if node.id not in checked_ids:
                    add_node(graph, node.id, node.name)
                    checked_ids.append(node.id)
                graph.add_edge(idx, node.id)

        graph.layout(prog='dot')
        graph.draw(path)
        return True
    else:
        logging.info('Cannot use pygraphviz, no figure saved here.')
        return False


def _draw_cnn_cell(graph, dag, start_index, prefix=''):
    """
    Draw one cell,
    :param graph:
    :param dag:
    :return:
    """
    checked_ids = [0 + start_index, 1 + start_index]
    output_ids = set([0 + start_index, 1 + start_index])
    add_node(graph, checked_ids[0], f'{prefix}_x[t-1]')
    add_node(graph, checked_ids[1], f'{prefix}_x[t]')
    graph.add_edge(checked_ids[0], checked_ids[1], label=f'prev_cell')
    num_nodes = 2
    for (source, target, op) in dag:
        target += start_index
        source += start_index

        if source in output_ids:
            output_ids.remove(source)

        if target not in checked_ids:
            checked_ids.append(target)
            add_node(graph, target, str(f'{prefix}_Node {target-1}'))
            num_nodes += 1
            output_ids.add(target)

        graph.add_edge(source, target, label=f'{op}')

    add_node(graph, num_nodes + start_index, f'concat')

    for idx in output_ids:
        graph.add_edge(idx, num_nodes + start_index)
    num_nodes += 1

    return graph, num_nodes


def draw_network_cnn(dag, path):
    makedirs(os.path.dirname(path))
    if enable_graph:
        graph = pgv.AGraph(directed=True, strict=True,
                           fontname='Helvetica', arrowtype='open')  # not work?

        if 'MicroArchi' in str(type(dag)):
            graph, num_nodes = _draw_cnn_cell(graph, dag.normal_cell, start_index=0, prefix='normal')
            graph, num_nodes = _draw_cnn_cell(graph, dag.reduced_cell, start_index=num_nodes, prefix='reduced')

        graph.layout(prog='dot')
        graph.draw(path)
        return True
    else:
        logging.info('Cannot use pygraphviz, no figure saved here.')
        return False


def make_gif(paths, gif_path, max_frame=50, prefix=""):
    import imageio

    paths.sort()

    skip_frame = len(paths) // max_frame
    paths = paths[::skip_frame]

    images = [imageio.imread(path) for path in paths]
    max_h, max_w, max_c = np.max(
            np.array([image.shape for image in images]), 0)

    for idx, image in enumerate(images):
        h, w, c = image.shape
        blank = np.ones([max_h, max_w, max_c], dtype=np.uint8) * 255

        pivot_h, pivot_w = (max_h-h)//2, (max_w-w)//2
        blank[pivot_h:pivot_h+h,pivot_w:pivot_w+w,:c] = image

        images[idx] = blank

    try:
        images = [Image.fromarray(image) for image in images]
        draws = [ImageDraw.Draw(image) for image in images]
        font = ImageFont.truetype("assets/arial.ttf", 30)

        steps = [int(os.path.basename(path).rsplit('.', 1)[0].split('-')[1]) for path in paths]
        for step, draw in zip(steps, draws):
            draw.text((max_h//20, max_h//20),
                      f"{prefix}step: {format(step, ',d')}", (0, 0, 0), font=font)
    except IndexError:
        pass

    imageio.mimsave(gif_path, [np.array(img) for img in images], duration=0.5)


##########################
# Torch
##########################

def detach(h):
    if torch.__version__ >= "0.4.0":
        return h.detach()
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)

def get_variable(inputs, cuda=False, requires_grad=True, **kwargs):

    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if torch.__version__ >= '0.4.0':
        """
        Do not support 
        """
        if cuda:
            out = inputs.cuda()
        else:
            out = inputs
        if not requires_grad:
            out.requires_grad = False
    else:
        if cuda:
            out = Variable(inputs.cuda(), requires_grad=requires_grad, **kwargs)
        else:
            out = Variable(inputs, requires_grad=requires_grad, **kwargs)
    return out

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    if isinstance(data, DataLoader):
        data.batch_size = bsz
        return data

    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data


##########################
# ETC
##########################

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def prepare_dirs(args):
    """Sets the directories for the model, and creates those directories.
    Args:
        args: Parsed from `argparse` in the `config` module.
    """
    if args.load_path:
        if args.load_path.startswith(args.log_dir):
            args.model_dir = args.load_path
        else:
            if args.load_path.startswith(args.dataset):
                args.model_name = args.load_path
            else:
                args.model_name = "{}_{}".format(args.dataset, args.load_path)
    else:
        if args.comment is not None:
            args.model_name = f"{args.dataset}_{args.comment}"
        else:
            args.model_name = f"{args.dataset}_{get_time()}"

    if not hasattr(args, 'model_dir'):
        args.model_dir = os.path.join(args.log_dir, args.model_name)
        if os.path.exists(args.model_dir):
            print("Existing ! choose to overwrite ")
            # IPython.embed()
    args.data_path = os.path.join(args.data_dir, args.dataset)

    for path in [args.log_dir, args.data_dir, args.model_dir]:
        if not os.path.exists(path):
            makedirs(path)


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logging.info("[*] MODEL dir: %s" % args.model_dir)
    logging.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def makedirs(path):
    if not os.path.exists(path):
        logging.info("[*] Make directories : {}".format(path))
        os.makedirs(path)


def remove_file(path):
    if os.path.exists(path):
        logging.info("[*] Removed: {}".format(path))
        os.remove(path)


def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logging.info("[*] {} has backup: {}".format(path, new_path))