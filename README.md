# MolLVAE
Ladder VAE for molecular generation task. 

A project for Deep Generative Models course in 2020 spring.

## Dependency

Tested with:

```
python 3.6.9
molsets 0.3.1
torch 1.4.0
rdkit 2019.03.4
```

## Installation

#### Manually

Currently only support install manually:

1. Clone the repository:
```bash
git clone https://github.com/wxx07/MolLVAE.git
```

2. [Install RDKit](https://www.rdkit.org/docs/Install.html) for running tests.

3. Install `mollvae`:
```bash
cd MolLVAE/
pip install -e .
```

# Reproduce results

### Training

```bash
cd MolLVAE/
python mollvae/train.py --device <device> \
                        --model_save <path to trained model> \
                        --log_path <path to training log> \
                        --train_load data/train.csv \
                        --valid_load data/valid.csv \
                        --lr_n_restarts 20 \
                        --ratio 0.05 \
                        --emb_sz 256 \
                        --enc_hidden_size 256 \
                        --enc_num_layers 1 \
                        --dec_hid_sz 512 \
                        --dec_n_layer 2 \
                        --ladder_d_size 256 128 64 \
                        --ladder_z_size 16 8 4 \
                        --ladder_z2z_layer_size 8 16 \
                        --dropout 0.2
```

Train with hyperparameters obtained from random searching.



### Evaluation

* #### Test set reconstruction

Run:

```bash
python mollvae/tests/test_set_reconstruct.py --device <device> \
                           --test_load data/test.csv \
                           --n_enc_zs 1 1 1 --n_dec_xs 1 --gen_bsz 128 \
                           --emb_sz 256 \
                           --enc_hidden_size 256 \
                           --enc_num_layers 1 \
                           --dec_hid_sz 512 \
                           --dec_n_layer 2 \
                           --ladder_d_size 256 128 64 \
                           --ladder_z_size 16 8 4 \
                           --ladder_z2z_layer_size 8 16 \
                           --dropout 0.2 \
                           --model_load res/exp.best_hyp_combo97/model_195.pt
```

to get one-per-one reconstruction rate ( one input, one latent code, one decoding attempt)



Run:

```bash
python mollvae/tests/test_set_reconstruct.py --device cuda:0 \
                           --test_load data/test.csv \
                           --n_enc_zs 10 10 10 --n_dec_xs 10 --gen_bsz 128 \
                           --emb_sz 256 \
                           --enc_hidden_size 256 \
                           --enc_num_layers 1 \
                           --dec_hid_sz 512 \
                           --dec_n_layer 2 \
                           --ladder_d_size 256 128 64 \
                           --ladder_z_size 16 8 4 \
                           --ladder_z2z_layer_size 8 16 \
                           --dropout 0.2 \
                           --model_load res/exp.best_hyp_combo97/model_195.pt
```

to get 100-per-1 reconstruction rate ( one input, ten latent codes, ten decoding attempts for each latent code)



* #### Sampling from prior distribution

Run:

```bash
python mollvae/sample.py --device cuda:1 \
                         --sample_type prior \
                         --n_enc_zs 1000 1000 1000 --n_dec_xs 10 --gen_bsz 128 \
                         --emb_sz 256 \
                         --enc_hidden_size 256 \
                         --enc_num_layers 1 \
                         --dec_hid_sz 512 \
                         --dec_n_layer 2 \
                         --ladder_d_size 256 128 64 \
                         --ladder_z_size 16 8 4 \
                         --ladder_z2z_layer_size 8 16 \
                         --dropout 0.2 \
                         --model_load res/exp.best_hyp_combo97/model_195.pt \
                         --sample_save <path to save sampled smiles>
```

to sample at top z layer 1000 times and attempt to decode 10 times for each latent code.

Show validity and unique rate.



Run:

```bash
python mollvae/sample.py --device cuda:1 \
                         --sample_type prior \
                         --n_enc_zs 1000 100 10 --n_dec_xs 10 --gen_bsz 128 \
                         --emb_sz 256 \
                         --enc_hidden_size 256 \
                         --enc_num_layers 1 \
                         --dec_hid_sz 512 \
                         --dec_n_layer 2 \
                         --ladder_d_size 256 128 64 \
                         --ladder_z_size 16 8 4 \
                         --ladder_z2z_layer_size 8 16 \
                         --dropout 0.2 \
                         --model_load res/exp.best_hyp_combo97/model_195.pt \
                         --sample_save prior.10x10x10.dec_xs_10.csv
```

to sample at top z layer 10 times, sample at next highest z layer 10 times for each z_top and so on. Attempts to decode 10 times for each latent code.

Show validity and unique rate.



* #### Hierarchical control experiment

Run

```bash
python mollvae/sample.py --device cuda:0 \
                         --sample_type control_z \
                         --sample_layer <z layer index> \
                         --n_enc_zs <n_enc_zs> --n_dec_xs 10 --gen_bsz 128 \
                         --emb_sz 256 \
                         --enc_hidden_size 256 \
                         --enc_num_layers 1 \
                         --dec_hid_sz 512 \
                         --dec_n_layer 2 \
                         --ladder_d_size 256 128 64 \
                         --ladder_z_size 16 8 4 \
                         --ladder_z2z_layer_size 8 16 \
                         --dropout 0.2 \
                         --model_load res/exp.best_hyp_combo97/model_195.pt \
                         --sample_save <path to save sampled smiles>
```

to get molecules decode from latent codes varying at layer `sample_layer`

Similar pipeline to [addtt's](https://github.com/addtt/ladder-vae-pytorch#hierarchical-representations) except skip connection.