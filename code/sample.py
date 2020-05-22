import torch
import pandas as pd
import math
from code.model.model import LVAE
from code.dataset import DatasetSplit
from code.opt import get_parser
from code.utils import set_seed
import rdkit

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
test_split = DatasetSplit("test", r"C:\Users\ASUS\github\MolLVAE\MolLVAE\data\test.csv")
vocab = test_split._vocab
dec_n = 10


print('Load model...')
model = LVAE(vocab,config)
model.load_state_dict(torch.load(load_model_from))
model.to(device)
model.eval()

# prior sampling
print('sampling...')
print('----------------------------------------------------------------')
with torch.no_grad():
    print('sample 1000 mol from prior distribution')
    samp_smiles_1 = model.sample(1000,max_len=150,deterministic=True)
    total_1 = len(samp_smiles_1)
    #vaild check
    vaild_1 = 0
    print('vaild check')
    for i in samp_smiles_1:
        if rdkit.Chem.MolFromSmiles(i) != None:
            vaild_1 += 1
    print('vaild rate: {}'.format(vaild_1/total_1))

    #unique check
    print('unique check')
    data_1 = pd.DataFrame(samp_smiles_1,columns=['smiles'])
    count_1 = data_1.loc[:,'smiles'].value_counts()
    print('unique rate: {}'.format(count_1.size/total_1))
    print('number of mol :{}'.format(total_1))
    print('----------------------------------------------------------------')

    #each mol decode 10 times
    print('each mol decode {} times'.format(dec_n))
    z_mu_p = []
    z_log_var_p = []
    z_sample = []
    z_mu_p.append(torch.zeros(1000,model.z_size[-1]).to(model.device()))
    z_log_var_p.append(torch.zeros(1000, model.z_size[-1]).to(model.device()))
    z_sample.append(model.sample_z(z_mu_p[0], z_log_var_p[0]))
    _, _, z_sample = model.gen_top_down(z_sample, z_mu_p, z_log_var_p)
    cat_z = z_sample[0]
    for i in range(len(model.z_size) - 1):
        cat_z = torch.cat((cat_z, z_sample[i + 1]), 1)
    cat_z = cat_z.repeat_interleave(dec_n,dim=0)
    samp_smiles_2 = model.sample(total_1*dec_n,z_in=cat_z,concated=True,max_len=150,deterministic=True)
    total_2 = len(samp_smiles_2)
    # vaild check
    vaild_2 = 0
    print('vaild check')
    for i in samp_smiles_2:
        if rdkit.Chem.MolFromSmiles(i) != None:
            vaild_2 += 1
    print('vaild rate: {}'.format(vaild_2 / total_2))

    # unique check
    print('unique check')
    data_2 = pd.DataFrame(samp_smiles_2, columns=['smiles'])
    count_2 = data_2.loc[:, 'smiles'].value_counts()
    print('unique rate: {}'.format(count_2.size / total_2))
    print('number of mol :{}'.format(total_2))
    print('----------------------------------------------------------------')
# sampling
with torch.no_grad():
    #sample z
    print('sample 10 times at each level z')
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
                z_sample[i+1] = torch.cat((z_sample[i+1],z),dim=1)
        z_sample[i+1] = z_sample[i+1].view(-1,z_size[-2-i])
        z_sample_re.append(z_sample[i+1].repeat_interleave(int(math.pow(10,len(z_size)-i-2)),dim=0))
    z_sample = list(reversed(z_sample))
    z_sample_re = list(reversed(z_sample_re))
    for i in range(len(z_size)):
        z_sample_re[i].repeat_interleave(100,dim=0)

    #decoding
    samp_smiles_3 = model.sample(n_batch=int(math.pow(10,len(z_size))),z_in=z_sample_re,max_len=150,deterministic=True)
    total_3 = len(samp_smiles_3)
    #vaild check
    vaild_3 = 0
    print('vaild check')
    for i in samp_smiles_3:
        if rdkit.Chem.MolFromSmiles(i) != None:
            vaild_3 += 1
    print('vaild rate: {}'.format(vaild_3/total_3))

    #unique check
    print('unique check')
    data_3 = pd.DataFrame(samp_smiles_3,columns=['smiles'])
    count_3 = data_3.loc[:,'smiles'].value_counts()
    print('unique rate: {}'.format(count_3.size/total_3))
    print('number of mol :{}'.format(total_3))
    print('----------------------------------------------------------------')

    # each mol decode 10 times
    print('each mol decode {} times'.format(dec_n))
    z_in = []
    for i in z_sample_re:
        z_in.append(i.repeat_interleave(dec_n, dim=0))
    samp_smiles_4 = model.sample(n_batch=total_3*dec_n,z_in=z_in,max_len=150,deterministic=True)
    total_4 = len(samp_smiles_4)
    # vaild check
    vaild_4 = 0
    print('vaild check')
    for i in samp_smiles_4:
        if rdkit.Chem.MolFromSmiles(i) != None:
            vaild_4 += 1
    print('vaild rate: {}'.format(vaild_4 / total_4))

    # unique check
    print('unique check')
    data_4 = pd.DataFrame(samp_smiles_4, columns=['smiles'])
    count_4 = data_4.loc[:, 'smiles'].value_counts()
    print('unique rate: {}'.format(count_4.size / total_4))
    print('number of mol :{}'.format(total_4))