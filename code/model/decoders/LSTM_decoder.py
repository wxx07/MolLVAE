# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:54:56 2020

@author: Dell
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_decoder(nn.Module):
    def __init__(self, vocab, embed_size, latent_size,\
                 decoder_hidden_size, decoder_num_layers,\
                 dropout
                ):
        super(LSTM_decoder, self).__init__()
        
        ## embedding layer
        n_vocab, d_emb, self.pad = len(vocab.i2c), embed_size, vocab.pad
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        
        ## middle layer
        self.d_z = latent_size
        self.d_d_h = decoder_hidden_size
        self.map_z2hc = nn.Linear(self.d_z, self.d_d_h*2) # h_0, c_0 should be of same shape
        
        ## output layer
        self.decoder_fc = nn.Linear(self.d_d_h, n_vocab)
        
        ## LSTM layer
        self.n_layer = decoder_num_layers
        self.lstm = nn.LSTM(d_emb + self.d_z,\
                            self.d_d_h,\
                            num_layers=self.n_layer,\
                            batch_first=True,\
                            dropout=dropout if decoder_num_layers > 1 else 0 # dropout layer follows lstm layer
                           )
        
    def forward(self, batch, z):
        
        
        padded_x, lengths = batch
        
        x_emb = self.x_emb(padded_x) # bsz * padded_len * emb_sz
        
        ## combine latent code and tokens as input: tearch forcing
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1) # bsz * padded_len * lat_sz
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first = True)
        
        
        ## get h_0 and c_0 from z
        hc_0 = self.map_z2hc(z) # bsz * (2 x d_d_h)
        h_0, c_0 = hc_0[:,:self.d_d_h], hc_0[:,self.d_d_h:]
        h_0 = h_0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1) # n_layer * bsz * d_d_h
        c_0 = c_0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1) # n_layer * bsz * d_d_h
        
        
        output, _ = self.lstm(x_input, (h_0, c_0)) # retrun PackedSequence obj
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # inverse op to pack_padded_sequence

        
        yhat = self.decoder_fc(output) # bsz * padded_len * n_vocab ,leave softmax op to CE 

        
        recon_loss = F.cross_entropy(
            yhat[:, :-1].contiguous().view(-1, yhat.size(-1)),
            padded_x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )
        
        return recon_loss
    
    def count_params(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)