from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import ipdb
import IPython
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDarts(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 source_length,
                 emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 args=None
                 ):
        """

        :param layers:
        :param vocab_size:      input-size to this encoder. should match the sequence length.
        :param hidden_size:
        :param dropout:
        :param length:          Does not have a real meaning to the layer.
        :param source_length:   Does not have a real meaning to the layer.
        :param emb_size:        embedding size
        :param mlp_layers:      layers of MLP predictor
        :param mlp_hidden_size: size of MLP
        :param mlp_dropout:     dropout rate.
        """
        super(EncoderDarts, self).__init__()
        self.args = args
        # TODO, adopt the source length these hyperparameter to align with new string structure, but to be honest, it is quite easy.
        self.layers = layers
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.length = length
        self.source_length = source_length
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size
        assert self.hidden_size == self.emb_size, 'Encoder hidden size should match the embedding size.'
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)        # size = [72, 20, 96] [Bs, SeqLength, Embedding_Dim]
        embedded = self.dropout(embedded)
        if self.source_length != self.length:
            assert self.source_length % self.length == 0
            ratio = self.source_length // self.length
            embedded = embedded.view(-1, self.source_length // ratio, ratio * self.emb_size)
        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out
        encoder_hidden = hidden
        
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        arch_emb = out          # [72, 96]
        
        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)  # accuracy prediction.
        return encoder_outputs, encoder_hidden, arch_emb, predict_value
    
    def infer(self, x, predict_lambda, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb = torch.mean(new_encoder_outputs, dim=1)
        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb


Encoder_Nasbench = EncoderDarts

# class Encoder_Nasbench(Encoder):

#     def forward(self, x):
#         embedded = self.embedding(x)  # size = [72, 20, 96]
#         embedded = self.dropout(embedded)
#         if self.source_length != self.length:
#             assert self.source_length % self.length == 0
#             ratio = self.source_length // self.length
#             embedded = embedded.view(-1, self.source_length // ratio, ratio * self.emb_size)
#         out, hidden = self.rnn(embedded)
#         out = F.normalize(out, 2, dim=-1)
#         encoder_outputs = out
#         encoder_hidden = hidden

#         out = torch.mean(out, dim=1)
#         out = F.normalize(out, 2, dim=-1)
#         arch_emb = out  # [72, 96]

#         out = self.mlp(out)
#         out = self.regressor(out)
#         predict_value = torch.sigmoid(out)  # accuracy prediction.
#         # IPython.embed(header='Checking forward of encoder...')
#         return encoder_outputs, encoder_hidden, arch_emb, predict_value


class EncoderProxylessNAS(nn.Module):
    def __init__(self,
                 layers,
                 mlp_layers,
                 vocab_size,
                 hidden_size,
                 mlp_hidden_size,
                 dropout,
                 length,
                 source_length,
                 ):
        super(EncoderProxylessNAS, self).__init__()
        self.layers = layers
        self.mlp_layers = mlp_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout = dropout
        self.length = length
        self.source_length = source_length
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        if self.source_length != self.length:
            assert self.source_length % self.length == 0
            ratio = self.source_length // self.length
            self.rnn = nn.LSTM(self.hidden_size * ratio, self.hidden_size * ratio, self.layers, batch_first=True, dropout=dropout, bidirectional=True)
            self.out_proj = nn.Linear(self.hidden_size * ratio * 2, self.hidden_size, bias=True)
        else:
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout, bidirectional=True)
            self.out_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.mlp = nn.ModuleList([])
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.append(nn.Linear(self.hidden_size, self.mlp_hidden_size))
            elif i == self.mlp_layers - 1:
                self.mlp.append(nn.Linear(self.mlp_hidden_size, self.hidden_size))
            else:
                self.mlp.append(nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size))
        self.regressor = nn.Linear(self.hidden_size, 1)
    
    def forward_predictor(self, x):
        residual = x
        for i, mlp_layer in enumerate(self.mlp):
            x = mlp_layer(x)
            x = F.relu(x)
            if i != self.mlp_layers:
                x = F.dropout(x, self.dropout, training=self.training)
        x = (residual + x) * math.sqrt(0.5)
        x = self.regressor(x)
        predict_value = torch.sigmoid(x)
        return predict_value

    def forward(self, x):
        x = self.embedding(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.source_length != self.length:
            assert self.source_length % self.length == 0
            ratio = self.source_length // self.length
            x = x.view(-1, self.source_length // ratio, ratio * self.hidden_size)
        out, hidden = self.rnn(x)
        out = self.out_proj(out)
        out = (out + x) * math.sqrt(0.5)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out
        encoder_hidden = hidden
        
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        arch_emb = out
        
        residual = out
        for i, mlp_layer in enumerate(self.mlp):
            out = mlp_layer(out)
            out = F.relu(out)
            if i != self.mlp_layers:
                out = F.dropout(out, self.dropout, training=self.training)
        out = (residual + out) * math.sqrt(0.5)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value
    
    def infer(self, x, predict_lambda, direction='+'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb = torch.mean(new_encoder_outputs, dim=1)
        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        #new_arch_predict_value = self.forward_predictor(new_arch_emb)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb