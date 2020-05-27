import torch
from mollvae.model.model import LVAE
from mollvae.opt import get_parser
from mollvae.utils.utils import set_seed
from mollvae.utils.rdkit_utils import get_mol, disable_rdkit_log, enable_rdkit_log

import os, pickle

###config
parser = get_parser()
config = parser.parse_args()

### utils func

def vaild_check(sample_smiles):
    disable_rdkit_log()
    valid_smis = []
    for i in sample_smiles:
        if get_mol(i) != None:
            valid_smis.append(i)          
    vaild_rate = len(valid_smis) / len(sample_smiles)
    enable_rdkit_log()
    return vaild_rate, valid_smis

def unique_check(sample_smiles):
    """Check unique rate of valid smiles"""
    smis_unique = set(sample_smiles)
    unique_rate = len(smis_unique) / len(sample_smiles)
    
    return unique_rate, smis_unique

def prior_sampling(model, config):
    
    if len(set(config.n_enc_zs))==1:
        print("Sampling only from top z layer")
    else:
        print('Sampling multiple times at each level z')
    
    #decoding
    samp_smiles = model.sample(config.n_enc_zs,
                               max_len=config.max_len,
                               n_dec_times=config.n_dec_xs,
                               deterministic=False,
                               sample_type="prior")      
        
    return samp_smiles
    
    

    
def control_z_sampling(model, config):
    print(f"Sampling from No.{config.sample_layer} z layer...")
    samp_smiles = model.sample(config.n_enc_zs,
                               max_len=config.max_len,
                               n_dec_times=config.n_dec_xs,
                               deterministic=True,
                               sample_type="control_z",
                               sample_layer=config.sample_layer)

    return samp_smiles


### initialization
device = torch.device(config.device)
set_seed(config.seed)

vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/train_vocab.pkl")
with open(vocab_path, "rb") as fi:
    vocab =pickle.load(fi)
    
mol_dec_times = config.n_dec_xs

###load model
print(f'Load model from {config.model_load}...')
model = LVAE(vocab,config)
model.load_state_dict(torch.load(config.model_load))
model.to(device)
model.eval()
    
      
## do sampling and valid & unique check
if config.sample_type == "prior":
    samp_smiles = prior_sampling(model, config)
elif config.sample_type == "control_z":
    samp_smiles = control_z_sampling(model, config)
    
#vaild check
print('vaild check')
vaild_rate, valid_smis = vaild_check(samp_smiles)
print('vaild rate: {}%'.format(vaild_rate*100))

#unique check
print('unique check')
unique_rate, smiles_unique = unique_check(valid_smis)
print('unique rate: {}%'.format(unique_rate*100))
print('number of mol :{}'.format(len(samp_smiles)))

if config.sample_save:
    with open(f"{config.sample_save}", "w") as fo:
        fo.write("smiles\n")
        for s in smiles_unique:
            fo.write(f"{s}\n")

        
        
        
        
        
        
        
