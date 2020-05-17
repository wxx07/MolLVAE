# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:36:20 2020

@author: Olive

Srcipt for sampling and evaluation
"""

import sys
sys.path.append("/work01/home/wxxie/project/drug-gen/mollvae/MolLVAE/code")
from dataset import DatasetSplit
from opt import get_parser
from model.model import LVAE
from utils import set_seed

import torch

from tqdm import tqdm

from moses.utils import CircularBuffer


########## config

parser = get_parser()
config = parser.parse_args("--device cuda:0 \
                           --n_enc_zs 1 --n_dec_xs 1 --gen_bsz 128".split())

device = torch.device(config.device)
set_seed(config.seed)

#n_sample = config.n_sample
test_csv = config.test_load
load_model_from = "../res/exp/model_049.pt"

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

print("Loading trained model...")
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
            
            recon_seqs = model.sample(gen_bsz, max_len=max_len, z_in=samples)
            
            for k, (r_s,s) in enumerate(zip(recon_seqs, input_seqs)): # loop over mols
                if r_s == s:
                    success_cnt += 1


total_trials = len(test_dataloader) * n_latcode * n_dec_xs
print(f"Test set reconstruction rate: {1.*success_cnt / total_trials * 100}%")
    

########## Check test set reconstruction rate
########## v2

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
        
        recon_seqs = model.sample(gen_bsz, max_len=max_len, z_in=z, concated=True)
        
        success_cnt += sum(1 for r_s,s in zip(recon_seqs,input_seqs) if r_s==s)
        
        
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
    
        
        
        
        

        
        
            
            
            
            
        
        
        
        
        









