import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class LSTM_encoder(torch.nn.Module):
    def __init__(self,embed_layer,vocab,config):
        super(LSTM_encoder,self).__init__()
        self.sorted_seq = config.enc_sorted_seq
        self.pad = vocab.pad
        self.embedding = embed_layer
        self.lstm = nn.LSTM(config.emb_sz,config.enc_hidden_size,batch_first=True,
                            num_layers=config.enc_num_layers,bidirectional = config.enc_bidirectional,
                            dropout= config.dropout if config.enc_num_layers > 1 else 0)

    def forward(self,batch):
        if self.sorted_seq:
            seq,lengths = batch
            embeds = self.embedding(seq) # batch_size * padded_len * embed_size
            seq = pack_padded_sequence(embeds, lengths=list(lengths), batch_first=True)
            output, (h_n, c_n) = self.lstm(seq)
            output, _ = pad_packed_sequence(output,batch_first=True)
            h_n = h_n[-(1 + int(self.lstm.bidirectional)):]
            h = torch.cat(h_n.split(1), dim=-1).squeeze(0) # batch_size * enc_hidden_size * i (if enc_bidirectional = True,i=2;else i=1)
        else:
            # sort input sqeuence
            lengths = self.get_length(batch)
            _, idx_sort = torch.sort(torch.Tensor(lengths), dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            seq_sort = batch.index_select(0, idx_sort)
            lengths_sort = lengths.index_select(0, idx_sort)
            # forward
            embeds = self.embedding(seq_sort) # batch_size * padded_len * embed_size
            seq_sort = pack_padded_sequence(embeds,lengths=list(lengths_sort),batch_first=True)
            output,(h_n,c_n) = self.lstm(seq_sort)
            output,_ = pad_packed_sequence(output,batch_first=True)
            h_n = h_n[-(1+int(self.lstm.bidirectional)):]
            h_n = torch.cat(h_n.split(1), dim=-1).squeeze(0) # batch_size * enc_hidden_size * i (if enc_bidirectional = True,i=2;else i=1)
            #unsort output and h_n
            output = output.index_select(0, idx_unsort)
            h = h_n.index_select(0, idx_unsort)
        return output,h

    def get_length(self,x):
        length = []
        a = x.numpy()
        for i in range(x.size()[0]):
            length.append(len(a[i])-np.sum(a[i]==self.pad))
        return torch.tensor(length,dtype=torch.float32)
