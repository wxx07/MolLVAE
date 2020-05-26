import torch
import pandas as pd
import math
from code.model.model import LVAE
from code.dataset import DatasetSplit
from code.opt import get_parser
from code.utils import set_seed, get_mol
#import rdkit

###config
parser = get_parser()
config = parser.parse_args("--device cuda:0 \
                           --sample_type prior_sampling -- \
                           --n_enc_zs 10 --n_dec_xs 10 --gen_bsz 128 \
                           --emb_sz 256 \
                           --enc_hidden_size 256 \
                           --enc_num_layers 1 \
                           --dec_hid_sz 512 \
                           --dec_n_layer 2 \
                           --ladder_d_size 256 128 64 \
                           --ladder_z_size 16 8 4 \
                           --ladder_z2z_layer_size 8 16 \
                           --dropout 0.2 \
                           --model_load ../res/exp.best_hyp_combo97/model_195.pt".split())

### utils func

def vaild_check(sample_smiles):
    valid_smis = []
    for i in sample_smiles:
        if get_mol(i) != None:
            valid_smis.append(i)          
    vaild_rate = len(valid_smis) / len(sample_smiles)
    return vaild_rate, valid_smis

def unique_check(sample_smiles):
    """Check unique rate of valid smiles"""
    data = pd.DataFrame(sample_smiles, columns=['smiles'])
    count = data.loc[:, 'smiles'].value_counts()
    unique_rate = count.size/len(sample_smiles)
    return unique_rate

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
test_split = DatasetSplit("test", config.test_load)
vocab = test_split._vocab
mol_dec_times = config.n_dec_xs

###load model
print('Load model...')
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
unique_rate = unique_check(valid_smis)
print('unique rate: {}%'.format(unique_rate*100))
print('number of mol :{}'.format(len(samp_smiles)))
