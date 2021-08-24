import logging

import torch.nn as nn


class MixedVertex(nn.Module):

    @property
    def current_op(self):
        return self.ops[self.vertex_type]

    @property
    def current_proj_ops(self):
        # dynamic compute the current project ops rather than store the pointer.
        return [self.proj_ops[p] for p in self._current_proj_ops]

    @property
    def available_ops(self):
        """ return the available ops of the current vertex type. """
        return list(self.ops.keys())

    def __init__(self, input_size, output_size, vertex_type, do_projection=None, args=None,
                 curr_vtx_id=None, curr_cell_id=None):
        """ basic initialization, assign the references only. """
        super(MixedVertex, self).__init__()
        self.args = args
        self.vertex_type = vertex_type
        self.output_size = output_size
        self.input_size = input_size
        self.curr_vtx_id = curr_vtx_id
        self.curr_cell_id = curr_cell_id

        # initialize these ops
        self.proj_ops = None
        self._current_proj_ops = []
        self.ops = None

    """ abstract methods to be implemented """
    def forward(self, input, weight=None):
        raise NotImplementedError("Should be implemented by subclass.")

    def load_partial_parameters(self):
        """ TODO later """
        pass

    """ utility methods """

    def summary(self, vertex_id=None):
        """
        Summary this vertex, including nb channels and connection.
        Prefer output is:
        vertex {id}: [ id (channel), ... ] -> channel
        :return:
        """
        summary_str = f'vertex {vertex_id}:  ['
        for input_id in self._current_proj_ops:
            if input_id == 0:
                summary_str += f' 0 ({self.proj_ops[0].current_outsize})'
            else:
                summary_str += f' {input_id} ({self.proj_ops[input_id].channels})'
        summary_str += f'] - {self.vertex_type} -> {self.output_size}'
        logging.info(summary_str)
        return summary_str

    def change_vertex_type(self, input_size, output_size, vertex_type, proj_op_ids=None):
        """change vertex type
        This is a universal method.

        :param input_size:
        :param output_size:
        :param vertex_type:
        :param proj_op_ids: None by default, used in NASBench.
        :return:
        """

        if proj_op_ids:
            self._current_proj_ops = []
            for ind, (in_size, do_proj) in enumerate(zip(input_size, proj_op_ids)):
                if do_proj == 0: # Conv projection.
                    self.proj_ops[do_proj].change_size(in_size, output_size)
                else:   # Truncate projection
                    self.proj_ops[do_proj].channels = int(output_size)
                    # print("Truncate output size ", output_size)

                # VERY IMPORTANT update the current proj ops list
                self._current_proj_ops.append(do_proj)

        if vertex_type in self.available_ops:
            self.vertex_type = vertex_type
        else:
            raise ValueError("Update vertex_type error! Expected {} but got {}".format(
                self.oper.keys(), vertex_type
            ))

        self.output_size = output_size
        self.input_size = input_size

    def trainable_parameters(self, prefix='', recurse=True):
        # print(f"compute trainable parameters, {self.vertex_type}, {self.current_proj_ops}")
        pf_d = {f'ops.{self.vertex_type}': self.current_op}
        for proj_id in self._current_proj_ops:
            pf_d[f'proj_ops.{proj_id}'] = self.proj_ops[proj_id]

        for pf, d in pf_d.items():
            for k, p in d.named_parameters(prefix=prefix + '.' + pf, recurse=recurse):
                yield k, p