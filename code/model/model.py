import torch
import torch.nn as nn
import torch.nn.functional as F
from code.model.decoders.LSTM_decoder import LSTM_decoder
from code.model.encoders.LSTM_encoder import LSTM_encoder

class LVAE(torch.nn.Module):
    def __init__(self,vocab,config):
        super(LVAE,self).__init__()
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))
        self.d_size = config.ladder_d_size
        self.z_size = config.ladder_z_size
        self.z2z_layer_size = config.ladder_z2z_layer_size
        self.z_reverse_size = list(reversed(config.ladder_z_size))
        self.vocab = vocab
        # Word embeddings layer
        self.embedding = nn.Embedding(len(self.vocab.i2c),config.emb_sz,self.vocab.pad)

        # Encoder
        if config.enc_type == 'lstm':
            self.encoder = LSTM_encoder(self.embedding,self.vocab,config)
            self.ladder_input_size = config.enc_hidden_size * (1 + int(config.enc_bidirectional))
        else :
            raise ValueError(
                "Invalid encoder_type"
            )

        # Decoder
        if config.dec_type == 'lstm':
            self.decoder = LSTM_decoder(self.vocab,self.embedding,config,config.ladder_z_size[0])
        else:
            raise ValueError(
                "Invalid decoder_type"
            )

        #ladder
        self.top_down_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()

        #get a list contain bottom up layers,which used to generate z_mu_q_d and z_log_var_q_d
        for i in range(len(self.d_size)):
            if i == 0:
                self.bottom_up_layers.append(MLP(in_size=self.ladder_input_size,layer_size=self.d_size[i],out_size=self.z_size[i]))
            else:
                self.bottom_up_layers.append(MLP(in_size=self.d_size[i-1],layer_size=self.d_size[i],out_size=self.z_size[i]))

        #get a list contain top down layers,which used to generate z_mu_p and z_log_var_p
        for i in range(len(self.z_reverse_size)-1):
            self.top_down_layers.append(MLP(in_size=self.z_reverse_size[i],layer_size=self.z2z_layer_size[i],out_size=self.z_reverse_size[i+1]))

    def device(self):
        return next(self.parameters()).device

    def bottom_up(self,input):
        '''
        Do the bottom up step and get z_mu_q_d and z_log_var_q_d

        :param: input of ladder part,size=(batch_size * ladder_input_size)
        :return:two lists: z_mu_q_d ,z_log_var_q_d
        '''
        z_mu_q_d = []
        z_log_var_q_d = []
        for i in range(len(self.d_size)):
            if i == 0:
                nn, mu_q_d, log_var_q_d = self.bottom_up_layers[i](input)
                z_mu_q_d.append(mu_q_d)
                z_log_var_q_d.append(log_var_q_d)
            else:
                nn, mu_q_d, log_var_q_d = self.bottom_up_layers[i](nn)
                z_mu_q_d.append(mu_q_d)
                z_log_var_q_d.append(log_var_q_d)
        return z_mu_q_d, z_log_var_q_d # shape:[[batch_size * z_size[0]],[batch_size * z_size[1]],....,[batch_size * z_size[-1]]]

    def top_down(self, z_sample, z_mu_p, z_log_var_p):
        '''
        Do top down step and get z_mu_p,z_log_var_p, also return samples of each level z

        :param z_sample: only have the samples of top z
        :param z_mu_p: only have the z_mu_p of top z (mu = 0)
        :param z_log_var_p: only have the z_log_var_p of top z (var = 1)
        :return three list :z_mu_p,z_log_var_p,z_sample
        '''
        for i in range(len(self.z_reverse_size) - 1):
            _, mu_p, log_var_p = self.top_down_layers[i](z_sample[i])
            z_sample.append(self.sample_z(mu_p, log_var_p))
            z_mu_p.append(mu_p)
            z_log_var_p.append(log_var_p)
        #the shape of z_mu_p,z_log_var_p and z_sample after loop:[[batch_size * z_size[-1]],[batch_size * z_size[-2]],...[batch_size * z_size[0]]]
        return list(reversed(z_mu_p)), list(reversed(z_log_var_p)), list(reversed(z_sample))#reverse lists of z_mu_p,z_log_var_p and z_sample

    def sample_z(self, mu, log_var):
        '''
        Sampling z ~ p(z)= N(mu,var)
        :return: sample of z
        '''
        stddev = torch.exp(log_var) ** 0.5
        out = mu + stddev * torch.randn(mu.size()).to(self.device())
        return out

    def Gaussian_update(self, mu_q_d, log_var_q_d, mu_p, log_var_p):
        '''
        Combine mu_q_d,var_q_d and mu_p,var_p to generate mu_q,var_q
        :return two tensor: mu_q,var_q
        '''
        var_q_d,var_p = torch.exp(log_var_q_d),torch.exp(log_var_p)
        x = torch.pow(var_q_d, -1)
        y = torch.pow(var_p, -1)
        var = torch.pow(torch.add(x, y), -1)
        mu = torch.add(mu_q_d*x, mu_p*y) * var
        return mu, torch.log(var)

    def KL_loss(self,q_mu,q_log_var,p_mu,p_log_var):
        q_var, p_var = torch.exp(q_log_var),torch.exp(p_log_var)
        kl = 0.5*(p_log_var - q_log_var + q_var/p_var + torch.pow(torch.add(q_mu,-p_mu),2)/p_var -1)
        return kl.sum(1).mean()

    def forward_latent(self,input):
        # initialize required lists
        z_mu_p = []
        z_log_var_p = []
        z_sample = []
        z_mu_q = []
        z_log_var_q = []
        kl_loss = 0

        #do bottom up step and get z_mu_q_d, z_log_var_q_d
        z_mu_q_d, z_log_var_q_d = self.bottom_up(input)# [[batch_size * z_size[0]],[batch_size * z_size[1]],...,[batch_size * z_size[-1]]]

        #add mu_p,var_p and samples of top z to list
        z_mu_p.append(torch.zeros(z_mu_q_d[-1].size()).to(self.device())) #[[batch_size * z_size[-1]]]
        z_log_var_p.append(torch.zeros(z_log_var_q_d[-1].size()).to(self.device())) #[[batch_size * z_size[-1]]]
        z_sample.append(self.sample_z(z_mu_q_d[-1], z_log_var_q_d[-1])) #[[batch_size * z_size[-1]]]

        #do top down step and get z_mu_p, z_log_var_p,z_sample
        z_mu_p, z_log_var_p, z_sample = self.top_down(z_sample, z_mu_p, z_log_var_p)# [[batch_size * z_size[0]],[batch_size * z_size[1]],...,[batch_size * z_size[-1]]]

        #combine z_mu_q_d, z_log_var_q_d, z_mu_p, z_log_var_p to generate z_mu_q and z_log_var_q
        for i in range(len(self.z_size)-1):
            mu, log_var = self.Gaussian_update(z_mu_q_d[i], z_log_var_q_d[i], z_mu_p[i], z_log_var_p[i])
            z_mu_q.append(mu)
            z_log_var_q.append(log_var)
        z_mu_q.append(z_mu_q_d[-1])
        z_log_var_q.append(z_log_var_q_d[-1])
        # the shape of z_mu_q,z_log_var_q: [[batch_size * z_size[0]],[batch_size * z_size[1]],...,[batch_size * z_size[-1]]]

        #calculate KL_loss
        for i in range(len(self.z_size)):
            kl_loss = kl_loss + self.KL_loss(z_mu_q[i], z_log_var_q[i], z_mu_p[i], z_log_var_p[i])
        return z_sample[0],kl_loss

    def forward(self, batch):
        _,h = self.encoder(batch)
        z,KL_loss = self.forward_latent(h)
        recon_loss = self.decoder(batch,z)
        return KL_loss,recon_loss

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocab.ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def sample(self,n_batch,max_len=100,temp=1.0,z=None):
        '''
        Generating n_batch samples

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param temp: temperature of softmax
        :param z: (n_batch, ladder_z_size[-1]) of floats, vector of top z or None
        :return: list of tensors of strings, samples sequence x
        '''
        with torch.no_grad():
            # get samples of  z
            z_sample = []
            z_mu_p = []
            z_log_var_p =[]
            z_mu_p.append(torch.zeros(n_batch,self.z_size[-1]).to(self.device()))
            z_log_var_p.append(torch.zeros(n_batch, self.z_size[-1]).to(self.device()))
            if z is not None:
                z_sample.append(z.to(self.device()))
            else:
                z_sample.append(self.sample_z(z_mu_p[0], z_log_var_p[0]))
            _,_,z_sample = self.top_down(z_sample,z_mu_p,z_log_var_p)
            z = z_sample[0].unsqueeze(1) # n_batch * 1 * ladder_z_size[0]

            # inital values
            h = self.decoder.map_z2hc(z_sample[0]) # n_batch * dec_hid_sz *2
            h_0, c_0 = h[:,:self.decoder.d_d_h], h[:,self.decoder.d_d_h:] # n_batch * dec_hid_sz
            h_0 = h_0.unsqueeze(0).repeat(self.decoder.lstm.num_layers, 1, 1)  # dec_n_layer * n_batch * dec_hid_sz
            c_0 = c_0.unsqueeze(0).repeat(self.decoder.lstm.num_layers, 1, 1)  # dec_n_layer * n_batch * dec_hid_sz
            w = torch.tensor(self.bos).repeat(n_batch).to(self.device()) # n_batch
            x = torch.tensor([self.pad]).repeat(n_batch,max_len).to(self.device()) # n_batch * max_len

            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len]).to(self.device()).repeat(n_batch) # a tensor record the length of each molecule, size=n_batch
            eos_mask = torch.zeros(n_batch, dtype=torch.bool).to(self.device()) # a tensor indicate the molecular generation process is over or not,size=n_batch

            # generating cycle
            for i in range(1,max_len):
                x_emb = self.embedding(w).unsqueeze(1) # n_batch * 1 * embed_size
                x_input = torch.cat([x_emb, z], dim=-1) # n_batch * 1 * (embed_size + ladder_z_size[0])
                output, (h_0,c_0) = self.decoder.lstm(x_input, (h_0,c_0)) # output size : n_batch * 1 * dec_hid_sz
                y = self.decoder.decoder_fc(output.squeeze(1))
                y = F.softmax(y / temp, dim=-1) # n_batch * n_vocab

                w = torch.multinomial(y, 1)[:, 0] # input of next generate step, size=n_batch
                x[~eos_mask, i] = w[~eos_mask] # add generated atom to molecule
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1 # update end_pads
                eos_mask = eos_mask | i_eos_mask #update eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            return [self.tensor2string(i_x) for i_x in new_x]


class MLP(torch.nn.Module):
    def __init__(self,in_size,layer_size,out_size):
        super(MLP,self).__init__()
        self.layer1 = nn.Linear(in_size,layer_size)
        self.layer2 = nn.Linear(layer_size,layer_size)
        self.mu = nn.Linear(layer_size,out_size)
        self.var = nn.Linear(layer_size,out_size)

    def forward(self,input):
        layer1 = F.leaky_relu(self.layer1(input))
        layer2 = F.leaky_relu(self.layer2(layer1))
        mu = self.mu(layer2)
        var = F.softplus(self.var(layer2)) + 1e-8
        return layer2,mu,var
