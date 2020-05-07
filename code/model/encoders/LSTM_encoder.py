import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence

class LSTM_encoder(torch.nn.Module):
    def __init__(self,embed_layer,vocab,config):
        '''
        :param embed_size:  Input size of LSTM
        :param embed_layer: embedding layer
        :param hidden_size: Hidden size of LSTM
        :param num_layers: Number of recurrent layers.
        :param vocab: Vocabulary of embedding
        :param dropout: dropout layer follows lstm layer
        :param sorted_seq: If the input_seq has been sorted,set this param to True and give
                           the seq and lengths when forward function is called. Default: False
        :param bidirectional: If True, becomes a bidirectional LSTM. Default: False
        '''
        super(LSTM_encoder,self).__init__()
        self.sorted_seq = config.enc_sorted_seq
        self.pad = vocab.pad
        self.embedding = embed_layer
        self.lstm = nn.LSTM(config.embed_size,config.enc_hidden_size,
                            config.enc_num_layers,bidirectional = config.enc_bidirectional,
                            dropout= config.enc_dropout if config.enc_num_layers > 1 else 0)

    def forward(self,seq,lengths=None):
        if self.sorted_seq:
            embeds = [self.embedding(i) for i in seq]
            seq = pad_sequence(embeds, batch_first=True, padding_value=self.pad)
            seq = pack_padded_sequence(seq, lengths=list(lengths), batch_first=True,)
            output, (h_n, c_n) = self.lstm(seq)
            output, _ = pad_packed_sequence(output,batch_first=True)
            h_n = h_n[-(1 + int(self.lstm.bidirectional)):]
            h = torch.cat(h_n.split(1), dim=-1).squeeze(0)
        else:
            # sort input sqeuence
            lengths = self.get_length(seq)
            _, idx_sort = torch.sort(torch.Tensor(lengths), dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            seq_sort = seq.index_select(0, idx_sort)
            lengths_sort = lengths.index_select(0, idx_sort)
            # forward
            embeds = [self.embedding(i) for i in seq_sort]
            seq_sort = pad_sequence(embeds, batch_first=True, padding_value=self.pad)
            seq_sort = pack_padded_sequence(seq_sort,lengths=list(lengths_sort),batch_first=True)
            output,(h_n,c_n) = self.lstm(seq_sort)
            output,_ = pad_packed_sequence(output,batch_first=True)
            h_n = h_n[-(1+int(self.lstm.bidirectional)):]
            h_n = torch.cat(h_n.split(1), dim=-1).squeeze(0)
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

