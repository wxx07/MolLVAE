import torch
import pandas as pd
import math
from code.model.model import LVAE
from code.dataset import DatasetSplit
from code.opt import get_parser
from code.utils import set_seed
import rdkit
import tqdm
from functools import reduce

parser = get_parser()
config = parser.parse_args("--device cuda:0 \
                           --n_enc_zs 1 --n_dec_xs 1 --gen_bsz 128".split())
device = torch.device(config.device)
set_seed(config.seed)
test_split = DatasetSplit("test", r"C:\Users\ASUS\github\MolLVAE\MolLVAE\data\test.csv")
vocab = test_split._vocab
load_model_from = "../res/exp/model_049.pt"


print('Load model...')
model = LVAE(vocab,config)
model.load_state_dict(torch.load(load_model_from))
model.to(device)
model.eval()

# prior sampling
print('sampling...')
with torch.no_grad():
    samp_smiles_1 = model.sample(1000)
    total = len(samp_smiles_1)
    #vaild check
    vaild = 0
    print('vaild check')
    for i in samp_smiles_1:
        if rdkit.Chem.MolFromSmiles(i) != None:
            vaild += 1
    print('vaild rate: {}'.format(vaild/total))

    #unique check
    print('unique check')
    data_1 = pd.DataFrame(samp_smiles_1,columns=['smiles'])
    count_1 = data_1.loc[:,'smiles'].value_counts().value_counts()
    print('unique rate: {}'.format(count_1[1]/total))

#sampling
with torch.no_grad():
    #sample z
    z_size = model.z_size
    z_sample = []
    z_sample_re =[]
    mu = torch.zeros((10,z_size[-1])).to(device)
    log_var = torch.zeros((10,z_size[-1])).to(device)
    z_sample.append(model.sample_z(mu,log_var))
    z_sample_re.append(z_sample[0].repeat_interleave(int(math.pow(10,len(z_size)-1)),dim=0))
    for i in range(len(z_size)-1):
        _,mu,log_var = model.top_down_layers[i](z_sample[i])
        for j in range(10):
            if j == 0 :
                z = model.sample_z(mu,log_var)
                z_sample.append(z)
            else:
                z = model.sample_z(mu, log_var)
                z_sample[i+1] = torch.cat((z_sample[i+1],z))
        z_sample[i+1] = z_sample[i+1].view(-1,z_size[-2-i])
        z_sample_re.append(z_sample[i+1].repeat_interleave(int(math.pow(10,len(z_size)-i-2)),dim=0))
    z_sample = list(reversed(z_sample))
    z_sample_re = list(reversed(z_sample_re))

    #decoding
    samp_smiles_2 = model.sample(n_batch=int(math.pow(10,len(z_size))),z_in=z_sample_re)
    total = len(samp_smiles_2)
    print(total)
    #vaild check
    vaild = 0
    print('vaild check')
    for i in samp_smiles_2:
        if rdkit.Chem.MolFromSmiles(i) != None:
            vaild += 1
    print('vaild rate: {}'.format(vaild/total))

    #unique check
    print('unique check')
    data_2 = pd.DataFrame(samp_smiles_2,columns=['smiles'])
    count_2 = data_2.loc[:,'smiles'].value_counts().value_counts()
    print('unique rate: {}'.format(count_2[1]/total))
