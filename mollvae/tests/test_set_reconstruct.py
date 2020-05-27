"""

Srcipt for sampling and evaluation
"""

from mollvae.dataset import DatasetSplit
from mollvae.opt import get_parser
from mollvae.model.model import LVAE
from mollvae.utils.utils import set_seed

import torch

from tqdm import tqdm
import numpy as np

#from moses.utils import CircularBuffer

########## config

parser = get_parser()
config = parser.parse_args()

device = torch.device(config.device)
set_seed(config.seed)

#n_sample = config.n_sample
test_csv = config.test_load

n_latcode = config.n_enc_zs[0] # get n_latcode latent codes for each mol
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

print(f"Loading trained model from {config.model_load}")
model = LVAE(vocab, config)
model.load_state_dict(torch.load(config.model_load))
model.to(device)
model.eval()
    

########## Check test set reconstruction rate
########## v2: one-per-one reconstruction rate

if n_dec_xs == 1:
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
            
            recon_seqs = model.sample([len(input_seqs)], max_len=max_len, z_in=z, concated=True)
            
            success_cnt += sum(1 for r_s,s in zip(recon_seqs,input_seqs) if r_s==s)
            
    total_trials = len(test_split.split_dataset.data) * n_latcode * n_dec_xs
    print(f"Test set reconstruction rate: {1.*success_cnt / total_trials * 100}%") 
        
########## Check test set reconstruction rate
########## v3: 1-per-100 reconstruction rate

else: 
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
            h = h.repeat_interleave(config.n_enc_zs[0], dim=0)
            
            z,_ = model.forward_latent(h) # z already concated
            z = z.repeat_interleave(config.n_dec_xs, dim=0)
            
            recon_seqs = model.sample([len(input_seqs)*config.n_enc_zs[0]*config.n_dec_xs],
                                      max_len=max_len, z_in=z, concated=True)
            
            input_seqs = list(np.repeat(input_seqs, config.n_enc_zs[0]*n_dec_xs))
            
            success_cnt += sum(1 for r_s,s in zip(recon_seqs,input_seqs) if r_s==s)
            
    total_trials = len(test_split.split_dataset.data) * n_latcode * n_dec_xs
    print(f"Test set reconstruction rate: {1.*success_cnt / total_trials * 100}%")        
        
########## check recon_loss for test set

#recon_loss_values = CircularBuffer(config.loss_buf_sz)
#data = tqdm(test_dataloader)
#for batch in data:
#    
#    with torch.no_grad():      
#        
#        batch = (batch[0].to(device), batch[1])
#        kl_loss, recon_loss = model(batch)
#        
#        recon_loss_values.add(recon_loss.item())
#        recon_loss_value = recon_loss_values.mean()
#        
#        postfix = [f'recon={recon_loss_value:.5f})']
#        data.set_postfix_str(' '.join(postfix))
        

    
        
        
        
        

        
        
            
            
            
            
        
        
        
        
        









