# -*- coding: utf-8 -*-
"""
Created on Sun May  3 23:25:59 2020

@author: Dell
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle,os

from moses import CharVocab

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class StringDataset:
    def __init__(self, vocab, data):
        """
        Creates a convenient Dataset with SMILES tokinization
        Arguments:
            vocab: CharVocab instance for tokenization
            data (list): SMILES strings for the dataset
        """
        self.vocab = vocab
        self.tokens = [vocab.string2ids(s) for s in data]
        self.data = data
        self.bos = vocab.bos
        self.eos = vocab.eos

    def __len__(self):
        """
        Computes a number of objects in the dataset
        """
        return len(self.tokens)

    def __getitem__(self, index):
        """
        Prepares torch tensors with a given SMILES.
        Arguments:
            index (int): index of SMILES in the original dataset
        Returns:
            A tuple (with_bos, with_eos, smiles), where
            * with_bos is a torch.long tensor of SMILES tokens with
                BOS (beginning of a sentence) token
            * with_eos is a torch.long tensor of SMILES tokens with
                EOS (end of a sentence) token
            * smiles is an original SMILES from the dataset
        """
        tokens = self.tokens[index]
        
        with_bos_eos = torch.tensor([self.bos] + tokens + [self.eos], dtype=torch.long)
        
        return with_bos_eos, self.data[index]

    def default_collate(self, batch, return_data=False):
        """
        Simple collate function for SMILES dataset. Joins a
        batch of objects from StringDataset into a batch
        Arguments:
            batch: list of objects from StringDataset
            pad: padding symbol, usually equals to vocab.pad
            return_data: if True, will return SMILES used in a batch
        Returns:
            with_bos, with_eos, lengths [, data] where
            * with_bos: padded sequence with BOS in the beginning
            * with_eos: padded sequence with EOS in the end
            * lengths: array with SMILES lengths in the batch
            * data: SMILES in the batch
        Note: output batch is sorted with respect to SMILES lengths in
            decreasing order, since this is a default format for torch
            RNN implementations
        """
        
        with_bos_eos, data = list(zip(*batch)) # data is list of clean smiles
        lengths = [len(x) for x in with_bos_eos] # count length should include <bos> and <eos>
        
        order = np.argsort(lengths)[::-1]
        with_bos_eos = [with_bos_eos[i] for i in order]
        lengths = [lengths[i] for i in order]
        
        with_bos_eos = torch.nn.utils.rnn.pad_sequence(
            with_bos_eos, padding_value=self.vocab.pad,\
            batch_first=True
        )

        if return_data:
            data = np.array(data)[order]
            return with_bos_eos, lengths, data
        
        return with_bos_eos, lengths

 
    

class DatasetSplit:
    """
    Class for getting training, validation and test dataset with dataloader
        
    Args:
        split: [str], split of dataset
        file_path: [str], file path to csv file containing SMILES
        
    Usage:
        >>> test_split = DatasetSplit("test","../data/test.csv")
        >>> test_dataloader = test_split.get_dataloader(batch_size=512)
        >>> for i,batch in enumerate(test_dataloader):
                ...

    
    """
    
    
    def __init__(self, split, file_path):
        assert split in ["train","valid","test"], "Unknown split called"
        self.split = split
        
        self.file_path = file_path
        split_raw = self.get_smiles()
        
        ## load vocab generated from training set
        train_vocab_path = base_dir + "/data/train_vocab.pkl"
        if split=="train" and not os.path.exists(train_vocab_path):
            self._vocab = CharVocab.from_data(split_raw) # [to be done] load from train_vocab
            with open(train_vocab_path, "wb") as fo:
                pickle.dump(self._vocab, fo)
        elif split in ["valid", "test"]:
            assert os.path.exists(train_vocab_path), "Dont have train_vocab"
            with open(train_vocab_path,"rb") as fi:
                self._vocab = pickle.load(fi)
            
        
        self.vocab = self._vocab.i2c
        
        self.split_dataset = StringDataset(self._vocab, split_raw)      
        
    def get_smiles(self):
        
        return pd.read_csv(self.file_path, usecols=["smiles"], squeeze=True).astype(str).tolist()
        
    
    def get_dataloader(self, batch_size=512):
        """
        Return:
            torch.utils.data.Dataloader: [object], loop over shuffled batches  
                                         and automatically reset, directly use in next epoch.
                                         
        Note:
            * a sample: (torch.tensor:1d, int)
                        torch.tensor: Tensor of ids of <bos>+SMILES+<eos>+ <pad>s
                        int: the length of above tensor BEFORE padding
            * a batch: (torch.tensor:`padded_length`x`batch_size`, list:int)
            * padded length (batch.size(0)) for batches is varying
        
        """
        
        return DataLoader(
                        self.split_dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=self.split_dataset.default_collate
                        )# reuse from epoch to epoch
        
    
