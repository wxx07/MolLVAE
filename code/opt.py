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
    model_args.add_argument("--enc_hidden_size",
                            type=int, default=256,
                            help="Encoder hidden vector size")
    model_args.add_argument("--enc_num_layers",
                            type=int, default=1,
                            help="Encoder number of lstm layers")
    model_args.add_argument("--enc_sorted_seq",
                            type=bool, default=True,
                            help="If the input of encoder is not sorted,set Flase")
    model_args.add_argument("--enc_bidirectional",
                            type=bool, default=False,
                            help="If True, becomes a bidirectional LSTM_encoder")
    
    ## decoder
    model_args.add_argument("--dec_hid_sz",
                            type=int, default=256,
                            help="Decoder hidden vector size")
    model_args.add_argument("--dec_n_layer",
                            type=int, default=1,
                            help="Decoder number of lstm layers")
    
    ## ladder latent code
    model_args.add_argument("--ladder_d_size",
                            type=list, default=[512,256,128,64,32],
                            help="The dimension of each layer in deterministic upward")
    model_args.add_argument("--ladder_z_size",
                            type=list, default=[64,32,16,8,4],
                            help="The dimension of each level latent z")
    model_args.add_argument("--ladder_z2z_layer_size",
                            type=list, default=[8,16,32,64],
                            help="the z2z layer size in to down step")

    
    
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
    train_args.add_argument("--clip_grad",
                            type=int, default=50,
                            help="Clip gradients to this value")
    train_args.add_argument("--loss_buf_sz",
                            type=int, default=1000,
                            help="Buffer losses for the last m batches")
    train_args.add_argument("--log_path",
                            type=str,
                            help="/path/to/experiment/log.csv")
    
    
    ## cosine annealing lr with restart
    train_args.add_argument("--lr_anr_type",
                            type=str, choices=["SGDR", "const"],
                            default="const",
                            help='choose lr annealer in \
                            "cosine annealing with restart" | \
                            "constant lr" | \
                            ...')
    train_args.add_argument("--lr_start",
                            type=float, default=3*1e-4,
                            help="Initial and max lr in annealing")
    train_args.add_argument("--lr_end",
                            type=float, default=3*1e-4,
                            help="Final and min lr in annealing")
    train_args.add_argument("--lr_period",
                            type=int, default=10,
                            help="Epoches before next restart")
    train_args.add_argument("--lr_n_restarts",
                            type=int, default=10,
                            help="Times of restart in annealing")
    train_args.add_argument("--lr_mult_coeff",
                            type=int, default=1,
                            help="Mult coefficient for period increment")
    
    
    
    
    
    
    ## linear increasing KL loss weight annealer
    # loss = kl_weight * kl_loss + rec_loss
    # annealing startpoint not necessary at the first epoch 
    train_args.add_argument("--kl_anr_type",
                            type=str, choices=["linear_inc", "const"],
                            default="const",
                            help='choose kl annealer in \
                            "linear increasing" | \
                            "constant lr" | \
                            ...')
    train_args.add_argument("--kl_e_start",
                            type=int, default=1,
                            help="Epoch start increasing KL weight")
    train_args.add_argument("--kl_w_start",
                            type=float, default=1.,
                            help="Initial KL weight")
    train_args.add_argument("--kl_w_end",
                            type=float, default=1.,
                            help="Final KL weight")
    
    
    
    
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
    
    parser.add_argument("--model_save",
                        type=str,
                        help="/path/to/trained/model.pt")
    parser.add_argument("--save_frequency",
                        type=int, default=10,
                        help="Every n epoches to save")
    
    
    
    return parser
    
    
    
    
    
    
    
    
    