import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class LSTM_encoder(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,vocab_size,batch_size,
                 bidirectional=False):
        '''
        :param input_size:  Input size of LSTM
        :param hidden_size: Hidden size of LSTM
        :param num_layers: Number of recurrent layers.
        :param vocab_size: Vocabulary size of embedding size
        :param batch_size: Batch size
        :param bidirectional: If True, becomes a bidirectional LSTM. Default: False
        '''
        super(LSTM_encoder,self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size,input_size)
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional = bidirectional)

    def forward(self,seq):
        # sort input sqeuence
        lengths = self.get_length(seq)
        _, idx_sort = torch.sort(torch.Tensor(lengths), dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        seq_sort = seq.index_select(0, idx_sort)
        lengths_sort = lengths.index_select(0, idx_sort)
        # forward
        embeds = self.embedding(seq_sort)
        seq_sort = pack_padded_sequence(embeds,lengths=list(lengths_sort),batch_first=True)
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
            length.append(np.size(a[i].nonzero()))
        return torch.tensor(length,dtype=torch.float32)