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
                            type=bool, default=True,
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
                            type=int, default=[128,64,32],
                            nargs="+", 
                            help="The dimension of each layer in deterministic upward")
    model_args.add_argument("--ladder_z_size",
                            type=int, default=[16,8,4],
                            nargs="+",
                            help="The dimension of each level latent z")
    model_args.add_argument("--ladder_z2z_layer_size",
                            type=int, default=[8,16],
                            nargs="+",
                            help="The z2z layer size in top down step")

    
    
    ########## train
    train_args = parser.add_argument_group('training')
    
    train_args.add_argument("--dropout",
                            type=float, default=.1,
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
                            type=int, default=20,
                            help="Buffer losses for the last m batches")
#    train_args.add_argument("--train_from",
#                            type=str,
#                            help="Load trained model from (Replaced by model_load)")
    
    
    
    ## cosine annealing lr with restart
    train_args.add_argument("--lr_anr_type",
                            type=str, choices=["SGDR","const"],
                            default="SGDR",
                            help='choose lr annealer in \
                            "cosine annealing with restart" | \
                            "constant lr (for hyperparams searching)" | \
                            ...')
    train_args.add_argument("--lr_start",
                            type=float, default=3*1e-4,
                            help="Initial and max lr in annealing")
    train_args.add_argument("--lr_end",
                            type=float, default=1e-6,
                            help="Final and min lr in annealing")
    train_args.add_argument("--lr_period",
                            type=int, default=10,
                            help="Epoches before next restart")
    train_args.add_argument("--lr_n_restarts",
                            type=int, default=5,
                            help="Times of restart in annealing")
    train_args.add_argument("--lr_mult_coeff",
                            type=int, default=1,
                            help="Mult coefficient for period increment")
    
    
    
    
    ## KL loss annealer: loss = beta * kl_loss + rec_loss with beta increasing from 0
    
    train_args.add_argument("--kl_anr_type",
                            type=str, choices=["linear_inc","const","cyclic","expo"],
                            default="cyclic",
                            help='choose kl annealer in \
                            "linear increasing" | \
                            "constant lr" | \
                            "cyclical increasing" | \
                            ...')
    # linear/monotonic increasing annealer
    train_args.add_argument("--kl_e_start",
                            type=int, default=0,
                            help="Epoch start increasing KL weight")
    train_args.add_argument("--kl_w_start",
                            type=float, default=1e-4,
                            help="Initial KL weight")
    train_args.add_argument("--kl_w_end",
                            type=float, default=1e-3,
                            help="Final KL weight")
    # cyclical annealer
    train_args.add_argument("--kl_n_cycle",
                            type=int, default=1,
                            help="Cycles to repeat. Last cycle ends with `kl_w_end`")
    train_args.add_argument("--ratio",
                            type=float, default=0.2,
                            help="Propotion of a cycle is used for increasing beta")
    
    
    ########## sample
    sample_args = parser.add_argument_group('sampling')
    sample_args.add_argument("--sample_type",
                             type=str, choices=["control_z","prior"],
                             default="prior",
                             help="Sample from a certain z or all zs")
#    sample_args.add_argument("--n_sample",
#                            type=int, default=1000,
#                            help="Number of samples from prior distribution")
    sample_args.add_argument("--n_enc_zs",
                            type=int, default=[1000,1000,1000],
                            nargs="+",
                            help="Number of unique samples at each z layer (bottom z -> top z). \
                            For prior sampling, e.g. reversed([10,100,1000])")
    sample_args.add_argument("--n_dec_xs",
                            type=int, default=10,
                            help="n decoding attempts for each latent code")
    sample_args.add_argument("--gen_bsz",
                            type=int, default=128,
                            help="Batches when sampling in parallel")
    sample_args.add_argument("--max_len",
                            type=int, default=100,
                            help="Max length of sampled SMILES (includes bos and eos)")
    sample_args.add_argument("--sample_save",
                             type=str,
                             help="/dir/to/save/result")
    
    ## for control_z sampling
    sample_args.add_argument("--sample_layer",
                             type=int,
                             help="Index of z layer to control and sample")
    
    
    
    
    
    
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
                        type=str, default="../data/test.csv",
                        help="File path to test.csv")
    
    parser.add_argument("--model_save",
                        type=str,
                        help="/path/to/trained/model.pt")
    parser.add_argument("--save_frequency",
                        type=int, default=10,
                        help="Every n epoches to save")
    
    parser.add_argument("--log_path",
                        type=str,
                        help="/path/to/experiment/log.csv")
    
    parser.add_argument("--model_load",
                        type=str,
                        help="/path/to/trained/model.pt (continue training or eval)")

    
    
    return parser
    
    
    
    
    
    
    
    
    