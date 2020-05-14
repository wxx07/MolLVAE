# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:52:07 2020

@author: Dell

Script for training mollvae
"""


from model.model import MolLVAE
from dataset import DatasetSplit
from opt import get_parser

from moses.utils import CircularBuffer, Logger

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler

from tqdm import tqdm
import numpy as np
import random
import math

############ utils func and class

def get_trainable_params(model):
    
    return (p for p in model.parameters() if p.requires_grad)

def get_lr_annealer(optimizer, config):
    
    if config.lr_anr_type == "SGDR":
        return CosineAnnealingLRWithRestart(optimizer, config)
    else:
        raise ValueError("Invalid lr annealer type")

    
def get_kl_annealer(n_epoch, config):
    
    if config.kl_anr_type == "const":
        return KLAnnealer_mono(n_epoch, 0, config.kl_w_start, config.kl_w_start)
    
    elif config.kl_anr_type == "linear_inc":
        return KLAnnealer_mono(n_epoch, config.kl_e_start, config.kl_w_start, config.kl_w_end)
    
    elif config.kl_anr_type == "cyclic":
        return KLAnnealer_cyc(n_epoch, config.kl_w_start, config.kl_w_end, config.kl_n_cycle, config.ratio)
    
    elif config.kl_anr_type == "expo":
        return KLAnnealer_expo(n_epoch)
    
    else:
        raise ValueError("Invalid kl annealer type")
    
def get_n_epoch(config):
    
    if config.lr_anr_type == "SGDR":
        n_epoch = sum(config.lr_period * (config.lr_mult_coeff ** i)
            for i in range(config.lr_n_restarts))
        print(f"Using SGDR annealer. Will train {n_epoch} epoches.")
        return n_epoch
    else:
        raise ValueError("Invalid lr annealer type")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer, config):
        self.n_period = config.lr_period
        self.n_mult = config.lr_mult_coeff
        self.lr_end = config.lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    """
    https://github.com/haofuml/cyclical_annealing
    """
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L    
    
class KLAnnealer_mono:
    """
    Control KL loss weight to increase linearly
        from epoch n_0 to the last epoch 
        leaving epoch < n_0 with constant `w_start` weight.
        "mono" means monotonic, compare with cyclical annealing.
        
    Adapted from `moses`    
    """
    
    def __init__(self, n_epoch, e_start, w_start, w_end):
        self.i_start = e_start
        self.w_start = w_start
        self.w_max = w_end
        self.n_epoch = n_epoch
        

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        # min(i)=0
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc
    
class KLAnnealer_cyc:
    """
    Implementation of "Cyclical Annealing Schedule: \
    A Simple Approach to Mitigating KL Vanishing"
    https://arxiv.org/abs/1903.10145
    """
    
    def __init__(self, n_epoch, w_start=0.0, w_end=1.0,  n_cycle=4, ratio=0.5):
        self.kl_weights = frange_cycle_linear(n_epoch, \
                                              start=w_start, \
                                              stop=w_end, \
                                              n_cycle=n_cycle, \
                                              ratio=ratio)
        

    def __call__(self, i):
        
        # min(i)=0
        return self.kl_weights[i]

def get_expo_inc_klws(n_epoch, w_start=1e-4, w_end=1.0, base=2):
    
    klws = np.ones(n_epoch) * w_end
    for i in range(n_epoch):
        klw = w_start * (base ** i)
        if klw <= w_end:
            klws[i] = klw
        else:
            break
    
    return klws
    
class KLAnnealer_expo:
    """
    Exponential increasing kl weight
    """
    
    def __init__(self, n_epoch, w_start=1e-4, w_end=1.0, base=2):
        self.kl_weights = get_expo_inc_klws(n_epoch, \
                                              w_start=w_start, \
                                              w_end=w_end, \
                                              base=base)
        
    def __call__(self, i):
        
        # min(i)=0
        return self.kl_weights[i]
    

def train_epoch(model, epoch, data, kl_weight, optimizer=None):
    
    if optimizer is None:
        model.eval()
    else:
        model.train()
        
    kl_loss_values = CircularBuffer(config.loss_buf_sz)
    recon_loss_values = CircularBuffer(config.loss_buf_sz)
    loss_values = CircularBuffer(config.loss_buf_sz)    
    for input_batch in data:
        input_batch = (input_batch[0].to(model.device()), input_batch[1])
        
        ## forward
        kl_loss, recon_loss = model(input_batch)
        loss = kl_weight * kl_loss + recon_loss
        
        ## backward
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(get_trainable_params(model), config.clip_grad)
            optimizer.step()
            
        ## log with buffer: average losses of the last m batches
        kl_loss_values.add(kl_loss.item())
        recon_loss_values.add(recon_loss.item())
        loss_values.add(loss.item())
        lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else 0)
        ## print out progress
        kl_loss_value = kl_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()
        postfix = [f'loss={loss_value:.5f}',
                   f'(kl={kl_loss_value:.5f}',
                   f'recon={recon_loss_value:.5f})',
                   f'klw={kl_weight:.5f} lr={lr:.5f}']
        data.set_postfix_str(' '.join(postfix))
            
    ## return results for this epoch (for tensorboard)
    postfix = {
        'epoch': epoch,
        'kl_weight': kl_weight,
        'lr': lr,
        'kl_loss': kl_loss_value,
        'recon_loss': recon_loss_value,
        'loss': loss_value,
        'mode': 'Eval' if optimizer is None else 'Train'}

    return postfix
            
    
    

def train(model, config, train_dataloader, valid_dataloader=None, logger=None):
    
    device = model.device()
    
    ## get optimizer and annealer
    n_epoch = get_n_epoch(config)
    optimizer = torch.optim.Adam(get_trainable_params(model),
                                 lr=config.lr_start)
    lr_annealer = get_lr_annealer(optimizer, config)
    kl_annealer = get_kl_annealer(n_epoch, config)
    
    ## iterative training
    model.zero_grad()
    for epoch in range(n_epoch):
        
        tqdm_data = tqdm(train_dataloader,
                         desc='Training (epoch #{})'.format(epoch))
        
        ## training
        kl_weight = kl_annealer(epoch)
        postfix = train_epoch(model, epoch, tqdm_data, kl_weight, optimizer)
        
        if logger is not None:
            logger.append(postfix)
            logger.save(config.log_path)
        
        ## validation
        if valid_dataloader is not None:
            
            tqdm_data = tqdm(valid_dataloader,
                         desc='Validation (epoch #{})'.format(epoch))
            postfix = train_epoch(model, epoch, tqdm_data, kl_weight)
            
            if logger is not None:
                logger.append(postfix)
                logger.save(config.log_path)
        
        ## save model
        if (config.model_save is not None) and \
            (epoch % config.save_frequency == 0):
            model = model.to("cpu")
            torch.save(model.state_dict(),
                       config.model_save[:-3] + "_{:03d}.pt".format(epoch))
            model = model.to(device)
        
        lr_annealer.step()

        

        
        








############ config

parser = get_parser()
config = parser.parse_args()

device = torch.device(config.device)

set_seed(config.seed)






############ load training data

print("Loading training set...")
train_split = DatasetSplit("train", config.train_load)
train_dataloader = train_split.get_dataloader(batch_size=config.train_bsz)

if config.valid_load is not None:
    print("Loading validation set...")
    valid_split = DatasetSplit("valid", config.valid_load)
    valid_dataloader = valid_split.get_dataloader(batch_size=config.train_bsz, shuffle=False)
    
vocab = train_split._vocab


############ get model and train

print("Initializing model...")
model = MolLVAE(vocab, config).to(device)

## log training process to csv file
logger = Logger() if config.log_path is not None else None

print("Start training...")
train(model, config, train_dataloader, valid_dataloader, logger)






