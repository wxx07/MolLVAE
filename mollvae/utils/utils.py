# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:52:12 2020

@author: Olive

Scripts for utilities
"""

import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

