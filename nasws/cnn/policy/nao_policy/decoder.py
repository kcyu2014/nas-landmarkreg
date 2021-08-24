import logging
import ipdb
import IPython
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_ID = 0
EOS_ID = 0


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None
    
    def set_mask(self, mask):
        self.mask = mask
    
    def forward(self, input, source_hids):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)
        
        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)
        
        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1, self.output_dim)
        
        return output, attn


class DecoderDarts(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 encoder_length,
                 args=None,
                 ):
        """[summary]

        Parameters
        ----------
        layers : [type]
            [description]
        vocab_size : int
            Vocabulary, number of distinct value in seq, usually used to separate the ops and nodes notation 
            Example, in DARTS search space, arch is [node1, op1, node2, op2 ...], 
                            where node ~ [0, ... num_nodes + 2], op ~ [0, max_op_id]
                        to distinct the op and node, we will map the node as
                            node = node + 1 (i.e. vocab)
                            op = op + num_nodes + 2 (two input nodes)

        hidden_size : [type]
            [description]
        dropout : [type]
            [description]
        length : [type]
            [description]
        encoder_length : [type]
            [description]
        args : [type], optional
            [description], by default None
        """
        super(DecoderDarts, self).__init__()
        self.args = args
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length
        self.encoder_length = encoder_length
        self.vocab_size = vocab_size
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.init_input = None
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward_step(self, x, hidden, encoder_outputs):
        """Step function in the decoder.
        Note that, the output size = 1 in NASBench case, because we only have 1 cell to predict!!!

        Parameters
        ----------
        x : Tensor
            input to decoder, in NB example [BS, output_size]
        hidden : tuple with shape [1, Bs, 36], [1, Bs, 36]
            Hidden state 
        encoder_outputs : tensor
            # TODO, QUESTION should be decoder outptus?        
            Output of the encoder? Size [Bs, Seq Length, 36]

        Returns
        -------
        [type]
            [description]
        """
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.embedding(x) # BS, 1, 36
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output, attn = self.attention(output, encoder_outputs)
        # attn: BS, 1, Seq length. 
        # output: Bs, 1, 36 (Embedding size)
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_size, -1)
        # bs, out_size, class
        return predicted_softmax, hidden, attn
    
    def forward(self, x, encoder_hidden=None, encoder_outputs=None):
        """
        B = 72 here, but 72 is not existing anywhere...
        TODO findout, why vocab size = num_Node(5) + num_ops(5) + 2 = 12?
        :param x: size: [B, controller_decoder_length]
        :param encoder_hidden: tuple, size=2, each of them, [1, B, 96 = encoder_hidden_size]
        :param encoder_outputs: [B, 20, 96]
        :return:
        """
        ret_dict = dict()
        ret_dict[DecoderDarts.KEY_ATTN_SCORE] = list()
        if x is None:
            inference = True
        else:
            inference = False
        x, batch_size, length = self._validate_args(x, encoder_hidden, encoder_outputs)
        assert length == self.length, f'sanity check, decoder_length {self.length} must match input {length}.'
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)
        
        def decode(step, step_output, step_attn):
            """

            :param step: di, which is matching the Arch, if 0, then it is a index. else it is op.
            :param step_output:
            :param step_attn:
            :return:
            """

            decoder_outputs.append(step_output)
            ret_dict[DecoderDarts.KEY_ATTN_SCORE].append(step_attn)
            symbols = self.decoder_outputs_to_symbols(step, decoder_outputs)
            sequence_symbols.append(symbols)
            
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols
        
        decoder_input = x[:, 0].unsqueeze(1)
    
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output yields [B,1, length], which is the vocabulary size!
            # step_attn -> [B, 1, length]
            step_output = decoder_output.squeeze(1) # [B, length]
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols
        
        ret_dict[DecoderDarts.KEY_SEQUENCE] = sequence_symbols   # lenght = 41 x [72,1]
        ret_dict[DecoderDarts.KEY_LENGTH] = lengths.tolist()
        
        # IPython.embed(header='Checking forward of decoder...')
        return decoder_outputs, decoder_hidden, ret_dict
    
    def decoder_outputs_to_symbols(self, step, decoder_outputs):
        """ node (4) + 2 + op (7) = 13, so vocab is 14 as the first is discarded for some unknown reason..."""
        op_start_index = self.args.num_intermediate_nodes + 2 # 5 is number of node, 2 is previous input and input.
        if step % 2 == 0:  # sample index, should be in [1, index-1]
            # index = tmp % (self.args.num_intermediate_nodes * 4) // 4 + 3  
            index = step % (self.args.num_intermediate_nodes * 4) // 4 + 3  
            # so here is the core part. because decoder_outputs[-1] is always vocab_size, so basically, index
            symbols = decoder_outputs[-1][:, 1:index].topk(1)[1] + 1
        else:  # sample operation, should be in [7, 11]
            symbols = decoder_outputs[-1][:, op_start_index:].topk(1)[1] + op_start_index
        
        return symbols

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden
    
    def _validate_args(self, x, encoder_hidden, encoder_outputs):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        
        # inference batch size
        if x is None and encoder_hidden is None:
            batch_size = 1
        else:
            if x is not None:
                batch_size = x.size(0)
            else:
                batch_size = encoder_hidden[0].size(1)
        
        # set default input and max decoding length
        if x is None:
            x = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).cuda()
            max_length = self.length
        else:
            max_length = x.size(1)
        
        return x, batch_size, max_length
    
    def eval(self):
        return
    
    def infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden


class Decoder_Nasbench_old(DecoderDarts):

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 encoder_length,
                 args,
                 ):
        # super(DecoderDarts, self).__init__()
        # self.args = args
        # basically compute this to verify.
        self.num_ops = 3
        self.child_nodes = args.num_intermediate_nodes
        self.num_cells_to_search = 1  # only have one cell to search
        # because, we do not have -1 -2 as previous input, so
        self.num_inputs = 1
        verify_vocab_size = self.num_inputs + self.child_nodes + self.num_ops # 1 + 5 + 3 = 9
        assert verify_vocab_size == vocab_size, f'Vocab size {vocab_size} is computed by self.num_inputs + self.child_nodes + self.num_ops' \
            f'{verify_vocab_size}'
        assert length == 2 * self.child_nodes, f'length = {length} but should be 2 * {self.child_nodes}'
        super().__init__(
                layers,
                vocab_size,
                hidden_size,
                dropout,
                length,
                encoder_length,
                args)
    
    def decoder_outputs_to_symbols(self, step, decoder_outputs):

        op_start_index = self.args.child_nodes + self.num_inputs  # 5 is number of node, 1 is previous input and input.
        if step % 2 == 0:  # sample index, should be in [1, index-1]
            # because they have reduced cell and not reduced cell, //2 should be remove
            # so, if length == 10, your input is just [2,3,4,5,6]
            index = step // self.num_cells_to_search % 10 // 2 + self.num_inputs  # TODO, super hardcode, fix it.
            # so here is the core part. because decoder_outputs[-1] is always vocab_size, so basically, index
            symbols = decoder_outputs[-1][:, 1:index + 1].topk(1)[1] + 1
        else:
            # sample operation, should be in [6,7,8] , op_start_index = 6,
            symbols = decoder_outputs[-1][:, op_start_index:].topk(1)[1] + op_start_index

        return symbols


class DecoderProxylessNAS(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 ):
        super(DecoderProxylessNAS, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length
        self.vocab_size = vocab_size
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = dropout
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_hidden = self._init_state(encoder_hidden)
        # training
        if x is not None:
            bsz = x.size(0)
            tgt_len = x.size(1)
            x = self.embedding(x)
            x = F.dropout(x, self.dropout, training=self.training)
            residual = x
            x, hidden = self.rnn(x, decoder_hidden)
            x = (residual + x) * math.sqrt(0.5)
            residual = x
            x, _ = self.attention(x, encoder_outputs)
            x = (residual + x) * math.sqrt(0.5)
            predicted_softmax = F.log_softmax(self.out(x.view(-1, self.hidden_size)), dim=-1)
            predicted_softmax = predicted_softmax.view(bsz, tgt_len, -1)
            return predicted_softmax, None

        # inference
        assert x is None
        bsz = encoder_hidden[0].size(1)
        length = self.length
        decoder_input = encoder_hidden[0].new(bsz, 1).fill_(0).long()
        decoded_ids = encoder_hidden[0].new(bsz, 0).fill_(0).long()
                
        for i in range(length):
            x = self.embedding(decoder_input[:, i:i+1])
            x = F.dropout(x, self.dropout, training=self.training)
            residual = x
            x, decoder_hidden = self.rnn(x, decoder_hidden)
            x = (residual + x) * math.sqrt(0.5)
            residual = x
            x, _ = self.attention(x, encoder_outputs)
            x = (residual + x) * math.sqrt(0.5)
            output = self.out(x.squeeze(1))
            symbol = self.decode(i, output)
            decoded_ids = torch.cat((decoded_ids, symbol), axis=-1)
            decoder_input = torch.cat((decoder_input, symbol), axis=-1)
        
        return None, decoded_ids
    
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden
    
    def decode(self, step, output):
        if step % 4 == 0: # must be conv, cannot be zerolayer
            symbol = output[:, 1:-1].topk(1)[1] + 1
        else:
            symbol = output[:, 1:].topk(1)[1] + 1
        return symbol


class DecoderNASbench(DecoderProxylessNAS):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 args=None,
                 ):
        from .utils_for_nasbench import compute_op_ids
        super(DecoderNASbench, self).__init__(                 
            layers,
            vocab_size,
            hidden_size,
            dropout,
            length
        )
        self.args = args
        self.op_ids = compute_op_ids(args.num_intermediate_nodes + 2)
        self.num_ops = 3
        self.child_nodes = args.num_intermediate_nodes
        self.num_cells_to_search = 1  # only have one cell to search
        # because, we do not have -1 -2 as previous input, so
        self.num_inputs = 1
        verify_vocab_size = 3 + self.num_ops # 1 + 5 + 3 = 9
        assert verify_vocab_size == vocab_size, f'Vocab size {vocab_size} is computed by number of ops + 2' \
            f'{verify_vocab_size}'
    
    def decode(self, step, output):
        # op_start_index = 2  # 5 is number of node, 1 is previous input and input.
        if step not in self.op_ids:  # sample index, should be in [0, 1] indicates the switch on or off of give edge
            symbols = output[:, 1:3].topk(1)[1] + 1
        else:
            # sample operation, should be in [2,3,4], op_start_index = 6,
            symbols = output[:, 3:].topk(1)[1] + 3
        return symbols


class DecoderNASbench201(DecoderProxylessNAS):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 args=None,
                 ):
        super(DecoderNASbench201, self).__init__(                 
            layers,
            vocab_size,
            hidden_size,
            dropout,
            length
        )
        self.args = args
        self.num_ops = 5
        self.child_nodes = args.num_intermediate_nodes
        self.num_cells_to_search = 1  # only have one cell to search
        self.num_inputs = 1
        verify_vocab_size = 1 + self.num_ops 
        assert verify_vocab_size == vocab_size, f'Vocab size {vocab_size} is computed by number of ops + 2' \
            f'{verify_vocab_size}'
    
    def decode(self, step, output):
        return output[:, 1:].topk(1)[1] + 1
