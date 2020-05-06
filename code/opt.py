# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:34:01 2020

@author: Dell

Script for storage and parsing of hyperparameters
"""

import argparse

def get_parser(parser=None):
    
    if parser is None: #? save for subparser
        parser = argparse.ArgumentParser()
    
    ########## experiment
    expr_args = parser.add_argument_group("experiment")
    add_expr_parser(expr_args)
    
    
    ########## model
    model_args = parser.add_argument_group('Model')
    model_args.add_argument("--enc_type",
                            type=str, default="lstm",
                            help="Encoder type")
    model_args.add_argument("--dec_type",
                            type=str, default="lstm",
                            help="Decoder type")
    
    ## encoder
    model_args.add_argument("--emb_sz",
                            type=int, default=128,
                            help="Embedding size")
    
    ## decoder
    model_args.add_argument("--dec_hid_sz",
                            type=int, default=256,
                            help="Decoder hidden vector size")
    model_args.add_argument("--dec_n_layer",
                            type=int, default=1,
                            help="Decoder number of lstm layers")
    
    ## ladder latent code
    
    
    ########## train
    train_args = parser.add_argument_group('training')
    
    train_args.add_argument("--dropout",
                            type=int, default=.1,
                            help="Global dropout rate")
    train_args.add_argument("--train_bsz",
                            type=int, default=512,
                            help="Batch size for training")
    train_args.add_argument("--n_epoch",
                            type=int, default=100,
                            help="Training epoches")
    
    ## cosine annealing lr with restart
    train_args.add_argument("--lr_anr_type",
                            type=str, choices=["SGDR"],
                            help='choose lr annealer in \
                            "cosine annealing with restart" | \
                            ...')
    train_args.add_argument("--lr_start",
                            type=float, default=1e-3,
                            help="Max lr in annealing")
    
    return parser
    
    
    
def add_expr_parser(parser):
    
    """
    Share when training and sampling
    
    Note:
        * train_load required=False because of sharing
        
    """
    
    parser.add_argument("--device",
                        type=str, default="cpu",
                        help='Device to run: "cpu" or "cuda:<device number>"')
    
    parser.add_argument("--seed",
                        type=int, default=56)
    
    parser.add_argument("--train_load",
                        type=str, default="../data/train.csv",
                        help="File path to train.csv")
    
    parser.add_argument("--valid_load",
                        type=str, default="../data/valid.csv",
                        help="File path to valid.csv")
    
    parser.add_argument("--test_load",
                        type=str, default="../data/valid.csv",
                        help="File path to test.csv")
    
    return parser
    
    
    
    
    
    
    
    
    