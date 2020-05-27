"""

Srcipt for sampling and evaluation
"""

from mollvae.dataset import DatasetSplit
from mollvae.opt import get_parser
from mollvae.model.model import LVAE
from mollvae.utils import set_seed

import torch

from tqdm import tqdm
import numpy as np

from moses.utils import CircularBuffer

########## config

parser = get_parser()
config = parser.parse_args("--device cuda:0 \
                           --n_enc_zs 10 --n_dec_xs 10 --gen_bsz 128 \
                           --emb_sz 256 \
                           --enc_hidden_size 256 \
                           --enc_num_layers 1 \
                           --dec_hid_sz 512 \
                           --dec_n_layer 2 \
                           --ladder_d_size 256 128 64 \
                           --ladder_z_size 16 8 4 \
                           --ladder_z2z_layer_size 8 16 \
                           --dropout 0.2".split())
load_model_from = "../res/exp.best_hyp_combo97/model_195.pt"

device = torch.device(config.device)
set_seed(config.seed)

#n_sample = config.n_sample
test_csv = config.test_load

n_latcode = config.n_enc_zs # get n_latcode latent codes for each mol
n_dec_xs = config.n_dec_xs
gen_bsz = config.gen_bsz
max_len = config.max_len

########## utils

def tensor2string_ad(tensor, vocab):
    """ Adapted from model.tensor2string. Consider pad indx in tensor. """
    
    ids = tensor.tolist()
    if vocab.pad in ids:
        pad_idx = ids.index(vocab.pad)
        ids = ids[:pad_idx]
    string = vocab.ids2string(ids, rem_bos=True, rem_eos=True)
    return string


########## get data and load trained model

test_split = DatasetSplit("test", test_csv)
test_dataloader = test_split.get_dataloader(batch_size=gen_bsz, shuffle=False)

vocab = test_split._vocab

print(f"Loading trained model from {load_model_from}")
model = LVAE(vocab, config)
model.load_state_dict(torch.load(load_model_from))
model.to(device)
model.eval()


########## Check test set reconstruction rate
########## v1

success_cnt = 0
for batch in tqdm(test_dataloader):
    
    with torch.no_grad():
        ## Input SMILES for ref
        padded_x, _ = batch
        input_seqs = []
        for s in padded_x:
            input_seqs.append(tensor2string_ad(s, vocab))         
        
        batch = (batch[0].to(device), batch[1])
        
        _,h = model.encoder(batch)
        
        z_mu_q_d, z_log_var_q_d = model.bottom_up(h)
        
        ## sample n_latcode times in top ladder z
        qd_mu_top = z_mu_q_d[-1].unsqueeze(1).repeat(1, n_latcode, 1) # (bsz, n_latcode, z_size[-1])
        qd_logvar_top = z_log_var_q_d[-1].unsqueeze(1).repeat(1, n_latcode, 1)
        z_sample_top = model.sample_z(qd_mu_top, qd_logvar_top)
        
        
        
        for j in range(n_latcode):
            
            z_sample = []
            z_sample.append(z_sample_top[:,j,:])
            
            _,_,_,_,samples = model.top_down(z_mu_q_d,z_log_var_q_d, z_sample=z_sample, mode="eval")
            
            recon_seqs = model.sample(len(input_seqs), max_len=max_len, z_in=samples)
            
            for k, (r_s,s) in enumerate(zip(recon_seqs, input_seqs)): # loop over mols
                if r_s == s:
                    success_cnt += 1


total_trials = len(test_dataloader) * n_latcode * n_dec_xs
print(f"Test set reconstruction rate: {1.*success_cnt / total_trials * 100}%")
    

########## Check test set reconstruction rate
########## v2: one-per-one reconstruction rate

success_cnt = 0
for batch in tqdm(test_dataloader):
    
    with torch.no_grad():

        ## Input SMILES for ref
        padded_x, _ = batch
        input_seqs = []
        for s in padded_x:
            input_seqs.append(tensor2string_ad(s, vocab))         
    
        batch = (batch[0].to(device), batch[1])
        
        
        _,h = model.encoder(batch)
        z,_ = model.forward_latent(h) # z already concated
        
        recon_seqs = model.sample(len(input_seqs), max_len=max_len, z_in=z, concated=True)
        
        success_cnt += sum(1 for r_s,s in zip(recon_seqs,input_seqs) if r_s==s)
        
########## Check test set reconstruction rate
########## v3: 1-per-100 reconstruction rate

success_cnt = 0
for batch in tqdm(test_dataloader):
    
    with torch.no_grad():

        ## Input SMILES for ref
        padded_x, _ = batch
        input_seqs = []
        for s in padded_x:
            input_seqs.append(tensor2string_ad(s, vocab))         
    
        batch = (batch[0].to(device), batch[1])
        
        _,h = model.encoder(batch)
        h = h.repeat_interleave(config.n_enc_zs, dim=0)
        
        z,_ = model.forward_latent(h) # z already concated
        z = z.repeat_interleave(config.n_dec_xs, dim=0)
        
        recon_seqs = model.sample(len(input_seqs)*config.n_enc_zs*config.n_dec_xs,
                                  max_len=max_len, z_in=z, concated=True)
        
        input_seqs = list(np.repeat(input_seqs, config.n_enc_zs*n_dec_xs))
        
        success_cnt += sum(1 for r_s,s in zip(recon_seqs,input_seqs) if r_s==s)
        
total_trials = len(test_split.split_dataset.data) * n_latcode * n_dec_xs
print(f"Test set reconstruction rate: {1.*success_cnt / total_trials * 100}%")        
        
########## check recon_loss for test set

recon_loss_values = CircularBuffer(config.loss_buf_sz)
data = tqdm(test_dataloader)
for batch in data:
    
    with torch.no_grad():      
        
        batch = (batch[0].to(device), batch[1])
        kl_loss, recon_loss = model(batch)
        
        recon_loss_values.add(recon_loss.item())
        recon_loss_value = recon_loss_values.mean()
        
        postfix = [f'recon={recon_loss_value:.5f})']
        data.set_postfix_str(' '.join(postfix))
        

    
        
        
        
        

        
        
            
            
            
            
        
        
        
        
        









