import ipdb
import os
import logging

import IPython
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import EncoderDarts, Encoder_Nasbench, EncoderProxylessNAS
from .decoder import DecoderDarts, DecoderNASbench,  DecoderProxylessNAS
from .decoder import Decoder_Nasbench_old as Decoder_Nasbench


SOS_ID = 0
EOS_ID = 0

class NAOProxylessNAS(nn.Module):
    def __init__(self,
                 encoder_layers,
                 mlp_layers,
                 decoder_layers,
                 vocab_size,
                 hidden_size,
                 mlp_hidden_size,
                 dropout,
                 encoder_length,
                 source_length,
                 decoder_length,
                 ):
        super(NAOProxylessNAS, self).__init__()
        self.encoder = EncoderProxylessNAS(
            encoder_layers,
            mlp_layers,
            vocab_size,
            hidden_size,
            mlp_hidden_size,
            dropout,
            encoder_length,
            source_length,
        )
        self.decoder = DecoderProxylessNAS(
            decoder_layers,
            vocab_size,
            hidden_size,
            dropout,
            decoder_length,
        )

        self.flatten_parameters()
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, archs = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        return predict_value, decoder_outputs, archs
    
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='+'):
        """Generate the new architecture in the sequence format.

        Parameters
        ----------
        input_variable : [type]
            [description]
        predict_lambda : int, optional
            [description], by default 1
        direction : str, optional
            [description], by default '+'

        Returns
        -------
        list
            a number of new architectures, but this is in the sequence format, need to transform
        """
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, new_archs = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        return new_archs


class NAO_Darts(nn.Module):
    def __init__(self,
                 encoder_layers,
                 encoder_vocab_size,
                 encoder_hidden_size,
                 encoder_dropout,
                 encoder_length,
                 source_length,
                 encoder_emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 decoder_layers,
                 decoder_vocab_size,
                 decoder_hidden_size,
                 decoder_dropout,
                 decoder_length,
                 args=None,
                 ):
        super(NAO_Darts, self).__init__()
        self.encoder = EncoderDarts(
            encoder_layers,
            encoder_vocab_size,
            encoder_hidden_size,
            encoder_dropout,
            encoder_length,
            source_length,
            encoder_emb_size,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout,
        )
        self.decoder = DecoderDarts(
            decoder_layers,
            decoder_vocab_size,
            decoder_hidden_size,
            decoder_dropout,
            decoder_length,
            encoder_length,
            args
        )

        self.flatten_parameters()
    
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()
    
    def forward(self, input_variable, target_variable=None):
        # Input to encoder is so-called sequence.
        
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        # decoder_outputs 41 x [72,12], same as ret['sequence']
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        # decoder_outputs becomes []
        arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, arch
    
    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        new_arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_arch


class NAO_Nasbench_old(NAO_Darts):
    def __init__(self,
                 encoder_layers,
                 encoder_vocab_size,
                 encoder_hidden_size,
                 encoder_dropout,
                 encoder_length,
                 source_length,
                 encoder_emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 decoder_layers,
                 decoder_vocab_size,
                 decoder_hidden_size,
                 decoder_dropout,
                 decoder_length,
                 args=None
                 ):
    
        super(NAO_Darts, self).__init__()
        self.args = args

        self.encoder = Encoder_Nasbench(
            encoder_layers,
            encoder_vocab_size,
            encoder_hidden_size,
            encoder_dropout,
            encoder_length,
            source_length,
            encoder_emb_size,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout,
            args,
        )
        self.decoder = Decoder_Nasbench(
            decoder_layers,
            decoder_vocab_size,
            decoder_hidden_size,
            decoder_dropout,
            decoder_length,
            encoder_length,
            args
        )

        self.flatten_parameters()

class NAO_Nasbench(NAOProxylessNAS):
    
    def __init__(self,
                 encoder_layers,
                 mlp_layers,
                 decoder_layers,
                 vocab_size,
                 hidden_size,
                 mlp_hidden_size,
                 dropout,
                 encoder_length,
                 source_length,
                 decoder_length,
                 args,
                 ):

        super(NAOProxylessNAS, self).__init__()                 
        self.encoder = EncoderProxylessNAS(
            encoder_layers,
            mlp_layers,
            vocab_size,
            hidden_size,
            mlp_hidden_size,
            dropout,
            encoder_length,
            source_length,
        )
        self.decoder = DecoderNASbench(
            decoder_layers,
            vocab_size,
            hidden_size,
            dropout,
            decoder_length,
            args
        )
    

class NAO_Nasbench201(NAOProxylessNAS):
    
    def __init__(self,
                 encoder_layers,
                 mlp_layers,
                 decoder_layers,
                 vocab_size,
                 hidden_size,
                 mlp_hidden_size,
                 dropout,
                 encoder_length,
                 source_length,
                 decoder_length,
                 args,
                 ):

        super(NAOProxylessNAS, self).__init__()                 
        self.encoder = EncoderProxylessNAS(
            encoder_layers,
            mlp_layers,
            vocab_size,
            hidden_size,
            mlp_hidden_size,
            dropout,
            encoder_length,
            source_length,
        )
        self.decoder = DecoderNASbench(
            decoder_layers,
            vocab_size,
            hidden_size,
            dropout,
            decoder_length,
            args
        )
    

