# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:52:07 2020

@author: Dell

Script for training mollvae
"""

#import sys,os

from model.model import MolLVAE
from dataset import DatasetSplit
from opt import get_parser

from moses.vae.misc import CosineAnnealingLRWithRestart, KLAnnealer

import torch

############ utils func

def get_trainable_params(model):
    
    return (p for p in model.parameters() if p.requires_grad)

def get_lr_annealer(optimizer, config):
    
    if config.lr_anr_type == "SGDR":
        return CosineAnnealingLRWithRestart(optimizer, config)
    
def get_n_epoch():
    
    if config.lr_anr_type == "SGDR":
        pass
    else:
        return config.n_epoch


def train(model, config, train_dataloader, valid_dataloader=None):
    
    ## get optimizer and annealer
    optimizer = torch.optim.Adam(get_trainable_params(model),
                                 lr=config.lr_start)
    lr_annealer = get_lr_annealer(optimizer, config) #! [to be done] mechanism

    pass


def train_epoch():
    pass



############ config

parser = get_parser()
config = parser.parse_args()

device = torch.device(config.device)

#! [to be done] set random seed










############ load training data

train_split = DatasetSplit("train", config.train_load)
train_dataloader = train_split.get_dataloader(batch_size=config.train_bsz)

if config.valid_load is not None:
    valid_split = DatasetSplit("valid", config.valid_load)
    valid_dataloader = valid_split.get_dataloader(batch_size=config.train_bsz, shuffle=False)
    
vocab = train_split._vocab


############ get model and train

model = MolLVAE(vocab, config).to(device)

train(model, config, train_dataloader, valid_dataloader)

















