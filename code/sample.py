import torch
import pandas as pd
import math
from code.model.model import LVAE
from code.dataset import DatasetSplit
from code.opt import get_parser
from code.utils import set_seed
import rdkit

def vaild_check(sample_smiles):
    vaild = 0
    for i in sample_smiles:
        if rdkit.Chem.MolFromSmiles(i) != None:
            vaild += 1
    vaild_rate = vaild / len(sample_smiles)
    return vaild_rate

def unique_check(sample_smiles):
    data = pd.DataFrame(sample_smiles, columns=['smiles'])
    count = data.loc[:, 'smiles'].value_counts()
    unique_rate = count.size/len(sample_smiles)
    return unique_rate

###config
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
mol_dec_times = 10
dec_deter = False

###load model
print('Load model...')
model = LVAE(vocab,config)
model.load_state_dict(torch.load(load_model_from))
model.to(device)
model.eval()

### prior sampling
print('sampling...')
print('----------------------------------------------------------------')
with torch.no_grad():
    print('from prior distribution of top z layer')
    samp_smiles_1 = model.sample(1000,max_len=150,deterministic=dec_deter)
    total_1 = len(samp_smiles_1)

    #vaild check
    print('vaild check')
    vaild_rate_1 = vaild_check(samp_smiles_1)
    print('vaild rate: {}'.format(vaild_rate_1))

    #unique check
    print('unique check')
    unique_rate_1 = unique_check(samp_smiles_1)
    print('unique rate: {}'.format(unique_rate_1))
    print('number of mol :{}'.format(total_1))
    print('----------------------------------------------------------------')

    #each mol decode 10 times
    print('each mol decode {} times'.format(mol_dec_times))

    #sample z
    z_mu_p = []
    z_log_var_p = []
    z_sample = []
    z_mu_p.append(torch.zeros(1000,model.z_size[-1]).to(model.device()))
    z_log_var_p.append(torch.zeros(1000, model.z_size[-1]).to(model.device()))
    z_sample.append(model.sample_z(z_mu_p[0], z_log_var_p[0]))
    _, _, z_sample = model.gen_top_down(z_sample, z_mu_p, z_log_var_p)

    #cat sampled z
    cat_z = z_sample[0]
    for i in range(len(model.z_size) - 1):
        cat_z = torch.cat((cat_z, z_sample[i + 1]), 1)

    #repeat cat_z n times for geneartion
    cat_z = cat_z.repeat_interleave(mol_dec_times,dim=0)
    samp_smiles_2 = model.sample(total_1*mol_dec_times,z_in=cat_z,concated=True,max_len=150,deterministic=dec_deter)
    total_2 = len(samp_smiles_2)

    # vaild check
    print('vaild check')
    vaild_rate_2 = vaild_check(samp_smiles_2)
    print('vaild rate: {}'.format(vaild_rate_2))

    # unique check
    print('unique check')
    unique_rate_2 = unique_check(samp_smiles_2)
    print('unique rate: {}'.format(unique_rate_2))
    print('number of mol :{}'.format(total_2))
    print('----------------------------------------------------------------')

# sampling
with torch.no_grad():
    #sample z
    print('sample 10 times at each level z')
    #sample 10 times at each level z
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
    z_sample_re = list(reversed(z_sample_re))#list contain each level z,unconcatenated

    #decoding
    samp_smiles_3 = model.sample(n_batch=int(math.pow(10,len(z_size))),z_in=z_sample_re,max_len=150,deterministic=dec_deter)
    total_3 = len(samp_smiles_3)

    #vaild check
    print('vaild check')
    vaild_rate_3 = vaild_check(samp_smiles_3)
    print('vaild rate: {}'.format(vaild_rate_3))

    #unique check
    print('unique check')
    unique_rate_3 = unique_check(samp_smiles_3)
    print('unique rate: {}'.format(unique_rate_3))
    print('number of mol :{}'.format(total_3))
    print('----------------------------------------------------------------')

    # each mol decode 10 times
    print('each mol decode {} times'.format(mol_dec_times))
    z_in = []
    for i in z_sample_re:
        z_in.append(i.repeat_interleave(mol_dec_times, dim=0))
    samp_smiles_4 = model.sample(n_batch=total_3*mol_dec_times,z_in=z_in,max_len=150,deterministic=dec_deter)
    total_4 = len(samp_smiles_4)
    # vaild check
    print('vaild check')
    vaild_rate_4 = vaild_check(samp_smiles_4)
    print('vaild rate: {}'.format(vaild_rate_4))

    # unique check
    print('unique check')
    unique_rate_4 = unique_check(samp_smiles_4)
    print('unique rate: {}'.format(unique_rate_4))
    print('number of mol :{}'.format(total_4))