import logging
from collections import OrderedDict

import IPython
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampler import obtain_full_model_spec
from .lib import config as _config
from ..supernet import Supernet

from .nasbench_api_v2 import ModelSpec_v2
from .model import NasBenchCell, NasBenchNet
from .model_builder import compute_vertex_channels
from .operations import conv_bn_relu, nasbench_vertex_weight_sharing

NASBENCH_CONFIG = _config.build_config()


def model_spec_to_wsbn_ids(model_spec, initial=False):
    """ generate the WSBN id spec correspondign to the model spec. 
        initial = True, return the total number.
    """
    num = model_spec.matrix.shape[0]
    a = model_spec.matrix

    if initial:
        return [2 ** i - 1 for i in range(num+1)]

    def mat_to_id(a, l): return int(
        '0b' + ''.join([str(int(i)) for i in a[:l, l].tolist()]), 2)
    return [mat_to_id(a, i) for i in range(num+1)]


def compute_expand_channel_choices(in_channels, output_channels, args, templates):
    channel_choices_dict = {}
    for i in range(args.num_intermediate_nodes + 2):
        channel_choices_dict[i] = []

    total_counts = []

    for m in templates:
        c = compute_vertex_channels(in_channels, output_channels, m.matrix)
        for i in range(len(c)-1):
            channel_choices_dict[i].append(c[i])
        channel_choices_dict[args.num_intermediate_nodes + 1].append(c[-1])
        total_counts.append(c)

    for i, dist in channel_choices_dict.items():
        print(f'For node {i}: ')
        unique, counts = np.unique(np.array(dist), return_counts=True)
        channel_choices_dict[i] = unique.tolist()
        print(f'Total possible channel choices: {unique}')
        print(f'Choice count: {counts}')

    print(set(['-'.join([str(d) for d in c[1:-1]]) for c in total_counts]))
    return channel_choices_dict


class NasBenchCellSearch(NasBenchCell):
    # heavily support the weight sharing scheme in this layer.

    def __init__(self, input_channels, output_channels, model_spec=None, args=None,
                 max_num_layer=None):
    
        super(NasBenchCell, self).__init__()  # nn.Module.init()

        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels
        # choose the vertex type.
        if args.bn_type == 'wsbn':
            args.nasbenchnet_vertex_type = 'mixedvertex_wsbn'
        
        self.vertex_cls = nasbench_vertex_weight_sharing[args.nasbenchnet_vertex_type]

        # make a full connection to initialize this cell.
        model_spec = model_spec or obtain_full_model_spec(
            args.num_intermediate_nodes + 2)
        full_connection = 1 - np.tril(np.ones_like(model_spec.matrix))
        model_spec.matrix = full_connection

        # assign the cell id
        self.cell_id = args.tmp_layer_id // args.tmp_num_modules_per_stack
        args.tmp_layer_id += 1
        self.layer_idx = self.cell_id
        self.max_num_layer = args.layers
        logging.debug("create the cell id {}, total stacks {}".format(
            self.cell_id, args.tmp_layer_id))

        # initialize the graph
        self.change_model_spec(model_spec, initial=True)
        if self.has_skip:
            self.projection = conv_bn_relu(1, input_channels, output_channels)

    def forward_darts(self, inputs, weightss):
        x, ws = inputs, weightss
        # Buffers to save intermediate resu
        tmp_buffers = {"input": x,
                       'output': 0.0}

        # Add output : 0.0 to avoid potential problem for [input, output] case.
        # Traverse based on predefined topological sort

        for output_node, input_nodes in self.execution_order.items():
            # Interior vertices are summed, outputs are concatenated
            if output_node != "output":
                in_buffers = [tmp_buffers[node] for node in input_nodes]
                node_id = eval(output_node.split('_')[1]) - 1
                weights = ws[node_id]
                tmp_buffers[output_node] = self.vertex_ops[output_node].forward_darts(in_buffers, weights)
            else:
                # We reverse the order to match tensorflow order for concatenation
                tmp_buffers[output_node] = torch.cat(
                    [tmp_buffers[ob] for ob in input_nodes], 1
                )
        # Becasue DARTS never zeros, so we always has skip.
        if self.has_skip:
            tmp_buffers["output"] += self.projection(x)

        return tmp_buffers["output"]

    def summary(self):
        for ind, v in enumerate(self.vertex_ops.values()):
            # each vertex summarize itself.
            v.summary(ind)

    def change_model_spec(self, model_spec, initial=False, verbose=False):

        # Setup a graph structure to simplify our life
        dag = nx.from_numpy_matrix(
            model_spec.matrix, create_using=nx.DiGraph())

        node_labels = {}
        for i, op in enumerate(model_spec.ops):
            if op == "input" or op == "output":
                node_labels[i] = op
            else:
                node_labels[i] = "vertex_%d" % i

        dag = nx.relabel_nodes(dag, node_labels)

        # Resolve dependencies in graph
        self.execution_order = self._get_execution_order(dag)

        # Setup output_sizes for operations and assign vertex types
        out_shapes_list = compute_vertex_channels(
            self.input_channels, self.output_channels, model_spec.matrix
        )
        if verbose:
            logging.info('vertex channels %s', str(out_shapes_list))

        if initial:
            # generate the maximum possible channels.
            out_shapes_list = [self.input_channels, ] + \
                [self.output_channels, ] * (len(out_shapes_list) - 1)
            logging.info('vertex channels %s', str(out_shapes_list))

        out_shapes = {}
        vertex_types = {}

        for t, (shape, op) in enumerate(zip(out_shapes_list, model_spec.ops)):
            out_shapes[node_labels[t]] = shape
            vertex_types[node_labels[t]] = op

        self.dag = dag

        # Setup the operations
        if initial:
            self.vertex_ops = nn.ModuleDict()
            if self.args.dynamic_conv_method == 'expand':
                node_channel_choices = compute_expand_channel_choices(
                    self.input_channels, self.output_channels, self.args, self.args.nasbench101_template_specs)
                logging.info('vertex channels for each node in expansion: ')
                for k, v in node_channel_choices.items():
                    logging.info(f"\tNode {k}: {v}")
            else:
                node_channel_choices = {k: None for k in range(self.args.num_intermediate_nodes + 2)}
            
        for output_node, input_nodes in self.execution_order.items():
            if output_node == "output":
                continue

            # Setup all input shapes
            in_shapes = [out_shapes[node] for node in input_nodes]

            # Check if any of the inputs to the vertex comes form input to module
            is_input = [node == "input" for node in input_nodes]
            if initial:
                # Setup the operation
                output_node_id = int(output_node.split('vertex_')[1])
                
                self.vertex_ops[output_node] = self.vertex_cls(
                    in_shapes, out_shapes[output_node], vertex_types[output_node], is_input,
                    curr_vtx_id=output_node_id, args=self.args, curr_cell_id=self.cell_id,
                    dynamic_in_channel_choices=node_channel_choices[output_node_id], 
                    dynamic_out_channel_choices=node_channel_choices[output_node_id]
                )
            else:
                # get the input_nodes order, by [input, vertex_i]
                input_nodes_id = [0 if x == 'input' else int(
                    x.split('vertex_')[1]) for x in input_nodes]
                
                self.vertex_ops[output_node].change_vertex_type(
                    in_shapes, out_shapes[output_node], vertex_types[output_node],
                    input_nodes_id
                )

        # Handle skip connections to output
        self.has_skip = self.dag.has_edge("input", "output")
        if self.has_skip:
            # if len(self.execution_order['output']) > 1:
            self.execution_order["output"].remove("input")
            if len(self.execution_order['output']) == 0:
                del self.execution_order['output']

        # handle the shared matrix.

    def trainable_parameters(self, prefix='', recurse=True):
        for k, m in self.vertex_ops.items():
            if isinstance(m, self.vertex_cls):
                # print(k)
                # print(m)
                for k, p in m.trainable_parameters(f'{prefix}.vertex_ops.{k}', recurse):
                    yield k, p
        if self.has_skip:
            for k, p in self.projection.named_parameters(f'{prefix}.projection', recurse):
                yield k, p

    def to_stand_alone(self):
        """to stand-alone cells. we need to write a cell to test this.

        Returns
        -------
        [type]
            [description]

        Yields
        -------
        [type]
            [description]
        """
        pass



class NasBenchNetSearch(NasBenchNet, Supernet):
    
    def __init__(self, args, config=NASBENCH_CONFIG, cell_cls=NasBenchCellSearch):
        """
        Weight sharing nasbench version 1.

        :param input_channels:
        :param model_spec:
        :param config: defined in nasbench.config
        """
        input_channels = 3
        model_spec = args.model_spec

        # create this for assigning labels.
        # override this num_stacks
        config["num_modules_per_stack"] = args.layers
        config['stem_filter_size'] = args.init_channels
        # args.layers = config["num_stacks"]      # fixed since NASBench is fixed
        args.tmp_layer_id = 0
        args.tmp_num_modules_per_stack = config['num_modules_per_stack']
        
        # if args.supernet_cell_type == 'op_on_node':
            # cell_cls = NasBenchCellSearch
        # elif args.supernet_cell_type == 'op_on_edge':
        Supernet.__init__(self, args)
        NasBenchNet.__init__(self, input_channels, model_spec, config, cell_cls, args)

        self.dropout = nn.Dropout(args.global_dropout_rate)
        del args.tmp_layer_id
        del args.tmp_num_modules_per_stack

    def change_model_spec(self, model_spec):
        """ Change the model spec """
        # for cell in self.stacks.values():
        self._model_spec = model_spec
        for k, v in self.stacks.items():
            # print("change spec for {}".format(k))
            for kk, vv in v.items():
                if 'module' in kk:
                    vv.change_model_spec(model_spec)
        return self

    def summary(self):
        """
        Display the summary of a NasBenchSearch with respect to current model spec.
        :return:
        """
        # For stacks.
        for ind, stack in enumerate(self.stacks.values()):
            # because all stack module has the same architecture, only display the first module.
            logging.info(f'Stack {ind}: ')
            acell = stack['module0']
            acell.summary()

    def trainable_parameters(self):
        """
        :return: trainable parameters that will be used in stack.forward
        """
        for k, v in self.stacks.items():
            for kk, vv in v.items():
                prefix = f'stacks.{k}.{kk}'
                if hasattr(vv, 'trainable_parameters'):
                    yield vv.trainable_parameters(prefix=prefix)
                else:
                    yield vv.named_parameters(prefix)

    def forward(self, inputs):
        # override this to avoid any confusion
        return Supernet.forward(self, inputs)

    def forward_oneshot(self, inputs):
        out = self.stem(inputs)
        for stack in self.stacks.values():
            for module in stack.values():
                out = module(out)
        out = F.avg_pool2d(out, out.shape[2:]).view(out.shape[:2])

        # adding global dropout rate
        out = self.dropout(out)
        return self.dense(out), None

    def unused_modules_off(self):
        if self.ALWAYS_FULL_PARAMETERS:
            return
        self._unused_modules = {'op': [], 'proj': []}
        for m in self.redundant_modules:
            unused = {}
            unused_projs = {}
            involved_index = [m.vertex_type]
            involved_proj_index = m._current_proj_ops
            for k in m.ops.keys():
                if k not in involved_index:
                    unused[k] = m.ops[k]
                    m.ops[k] = None
            self._unused_modules['op'].append(unused)
            self._unused_modules['proj'].append(unused_projs)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused, unused_projs in zip(self.redundant_modules,
                             self._unused_modules['op'], 
                             self._unused_modules['proj']):
            for k, o in unused.items():
                m.ops[k] = o
            for k, o in unused_projs.items():
                m.proj_ops[k] = o
        self._unused_modules = None

    def to_stand_alone_net(self):
        """ TODO (4.28) let's do this.
        zip to stand-alone net and compare the results.
            ideally, this should used like this:

            >>> 
                model_search.change_genotype(arch)
                net1 = model_search.to_stand_alone_net()
                net2 = NasBenchNet(arch, args)
                compare_two_network(net1, net2) # Return True.

        Returns
        -------
            return the weights or return the network, can be both.
        """
        return None


class NasBenchNetSearchDARTS(NasBenchNetSearch):
    # we use the nasbench 1shot1 definition then, 
    # fix the search space as 
    # node 1,2,3 has 1 previous connection, 4, 5 has 2.
    # output has 2 connection. Basically this is done.
    
    def __init__(self,  args, config=NASBENCH_CONFIG, cell_cls=NasBenchCellSearch) -> None:
        # overwrite the model spec
        self.num_nodes = args.num_intermediate_nodes
        self.num_ops = 3
        node = self.num_nodes
        args.model_spec = obtain_full_model_spec(node + 2)
        super().__init__(args, config, cell_cls)

        matrix = 1 - np.tril(np.ones([node + 2, node + 2]))
        new_matrix = np.concatenate([np.zeros([1, node]), matrix[0:-2, 1:-1]], axis=0) * 1e-3
        alpha_topology = torch.tensor(new_matrix, dtype=torch.float32) * torch.randn(*new_matrix.shape)
        alpha_ops = 1e-3 * torch.randn(self.num_ops, new_matrix.shape[0])
        self.arch_parameters = [alpha_topology, alpha_ops]
        # initialize the model with a good example architecture to initialize the channels
    
    def forward_darts(self, inputs):
        aux_logits = None
        out = self.stem(inputs)
        
        weightss = [self.weights(i) for i in range(self.num_nodes)]
        for stack in self.stacks.values():
            for k, module in stack.items():
                if 'module' in k:
                    out = module.forward_darts(out, weightss)
                else:
                    out = module(out)

        out = F.avg_pool2d(out, out.shape[2:]).view(out.shape[:2])
        return self.dense(out), aux_logits

    def forward_softoneshot(self, inputs):
        aux_logits = None
        out = self.stem(inputs)
        weightss = self._compute_softoneshot_arch_parameters(self.model_spec_cache)
        for stack in self.stacks.values():
            for k, module in stack.items():
                if 'module' in k:
                    out = module.forward_darts(out, weightss)
                else:
                    out = module(out)

        out = F.avg_pool2d(out, out.shape[2:]).view(out.shape[:2])
        return self.dense(out), aux_logits
        

    def weights(self, node):
        """Sample weights for input node"""
        return self.topology_weights(node), self.ops_weights(node)

    def topology_weights(self, node):
        # return the soft weights for topology aggregation
        return nn.functional.softmax(self.arch_parameters[0][: node + 2, node], dim=0)[1:]

    def ops_weights(self, node):
        return nn.functional.softmax(self.arch_parameters[1][:, node], dim=0)

    def _compute_softoneshot_arch_parameters(self, model_spec):
        return 0

    def _sample_model_spec(self, num, sample_archs, sample_ops, method='argmax'):
        """ support function to sample one architecture """
        if method == 'argmax':
            sample_fn = lambda x: x.probs.argmax()
        elif method == 'random':
            sample_fn = lambda x: x.sample()
        else:
            raise ValueError('Only support argmax or random to sample')
        new_model_specs = []

        with torch.no_grad():
            for _ in range(num):
                new_matrix = np.zeros((self.num_intermediate_nodes + 2, self.num_intermediate_nodes + 2), dtype=np.int)
                new_ops = ['input', ] + [None, ] * self.num_intermediate_nodes + ['output']
                for i in range(self.num_intermediate_nodes):
                    # action = 0 means, sample drop path
                    action = sample_fn(sample_archs[i]) - 1
                    if -1 < action < i + 1:  # cannot sample current node id
                        new_matrix[action, i + 1] = 1
                    # sample ops
                    op_action = sample_fn(sample_ops[i])
                    new_ops[i + 1] = self.AVAILABLE_OPS[op_action]
                # logging.debug("Sampled architecture: matrix {}, ops {}".format(new_matrix, new_ops))
                new_matrix[:, -1] = 1  # connect all output
                new_matrix[-1, -1] = 0  # make it upper trianguler
                mspec = ModelSpec_v2(new_matrix, new_ops)
                # logging.debug('Sampled model spec {}'.format(mspec))
                new_model_specs.append(mspec)
                # mspec.hash_spec(self.AVAILABLE_OPS)
        return new_model_specs

    def genotype(self, method='argmax', num=1):
        """
        Sample model specs by number.
        :param num:
        :return: list, num x [architecture ]
        """
        alpha_topology = self.alpha_topology.detach().clone()
        alpha_ops = self.alpha_ops.detach().clone()
        sample_archs = []
        sample_ops = []
        with torch.no_grad():
            for i in range(self.num_intermediate_nodes):
                # align with topoligy weights
                probs = nn.functional.softmax(alpha_topology[: i+2, i], dim=0)
                sample_archs.append(torch.distributions.Categorical(probs))
                probs_op = nn.functional.softmax(alpha_ops[:, i], dim=0)
                sample_ops.append(torch.distributions.Categorical(probs_op))
            res = self._sample_model_spec(num, sample_archs, sample_ops, method)
            if num == 1:
                return res[0]
            return res


class NasBenchCellSearchSoftWS(NasBenchCellSearch):

    def trainable_parameters(self):
        """
        Trainable parameters here include soft-weight-alpha and the parameters.
        :return:
        """
        pass


class NasBenchNetSearchSoftWeightSharing(NasBenchNetSearch):
    # this is to implement the soft-weight-sharing cell.
    def __init__(self, args, config=NASBENCH_CONFIG, cell_cls=NasBenchCellSearchSoftWS):
        super(NasBenchNetSearchSoftWeightSharing, self).__init__()
