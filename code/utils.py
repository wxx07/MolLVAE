# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:52:12 2020

@author: Olive

Scripts for utilities
"""

import torch
import random
import numpy as np

from rdkit import Chem


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_mol(smiles):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles, str):
        if len(smiles) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    else:
        return None