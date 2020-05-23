import torch
import os
from code.model.model import LVAE
from code.dataset import DatasetSplit
from code.opt import get_parser
from code.utils import set_seed
import rdkit
from rdkit.Chem import Draw


def get_z(z_mu_p, z_log_var_p, z_sample, sample_layer, num_of_sample, layer_num):
    '''
    sample n times at special z layer

    :param z_mu_p: z_mu_p of each level z from top-down step
    :param z_log_var_p:z_log_var_p of each level z from top-down step
    :param z_sample:z_sample of each level z from top-down step
    :param sample_layer: the index of z layer needing sample multiple times
    :param num_of_sample: sample times of special z layer
    :param layer_num: number of z layers

    :return: a list contain sampled z at each level
    '''
    # repeat z layers(>sample_layer) n times
    for i in range(layer_num - sample_layer - 1):
        z_sample[-i - 1] = z_sample[-i - 1].repeat(num_of_sample, 1)

    # sample special z layer n times
    for i in range(num_of_sample):
        z = model.sample_z(z_mu_p[-layer_num + sample_layer], z_log_var_p[-layer_num + sample_layer])
        if i == 0:
            z_sample[-layer_num + sample_layer] = z
        else:
            z_sample[-layer_num + sample_layer] = torch.cat((z_sample[-layer_num + sample_layer], z), dim=0)

    # calculate mu of z layers(<sample_layer),which used as sampled z
    for i in range(sample_layer):
        for j in range(num_of_sample):
            if j == 0:
                _, mu, _ = model.top_down_layers[layer_num - sample_layer + i - 1](
                    z_sample[-layer_num + sample_layer - i][j])
                mu = mu.unsqueeze(0)
                z_sample[-layer_num + sample_layer - i - 1] = mu
            else:
                _, mu, _ = model.top_down_layers[layer_num - sample_layer + i - 1](
                    z_sample[-layer_num + sample_layer - i][j])
                mu = mu.unsqueeze(0)
                z_sample[-layer_num + sample_layer - i - 1] = torch.cat(
                    (z_sample[-layer_num + sample_layer - i - 1], mu), dim=0)
    return z_sample


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

###load model
print('Load model...')
model = LVAE(vocab,config)
model.load_state_dict(torch.load(load_model_from))
model.to(device)
model.eval()

# initial parameters
num_of_sample = 10
layer_num = len(model.z_size)
res_path = r'../res/control_z_exp/'


for i in range(layer_num):
    sample_layer = i
    dir_path = os.path.join(res_path, 'z_{}'.format(sample_layer))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with torch.no_grad():
        # get one sample from top down step
        z_mu_p = []
        z_log_var_p = []
        z_sample = []
        z_mu_p.append(torch.zeros(1, model.z_size[-1]).to(device))
        z_log_var_p.append(torch.zeros(1, model.z_size[-1]).to(device))
        z_sample.append(model.sample_z(z_mu_p[0], z_log_var_p[0]))
        z_mu_p,z_log_var_p,z_sample = model.gen_top_down(z_sample,z_mu_p,z_log_var_p)

        #sample n times at special z layer
        z_sample = get_z(z_mu_p,z_log_var_p,z_sample,sample_layer,num_of_sample,layer_num)
        z_save_path = os.path.join(dir_path, 'z_sample.txt')
        with open(z_save_path,'w') as f:
            for z in z_sample:
                print(z,file=f)
                print('----------------------------------------------------------------------------------------------------',file=f)
        mol = model.sample(n_batch=num_of_sample,max_len=150,z_in=z_sample,deterministic=True)

        #save img
        mol_list = []
        for j in mol:
            mol_list.append(rdkit.Chem.MolFromSmiles(j))
        img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(200, 200))
        img_save_path = os.path.join(dir_path, 'mol.png')
        img.save(img_save_path)