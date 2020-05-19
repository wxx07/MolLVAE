# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:47:10 2020

@author: Olive

Script for getting random hyperparams search combination
"""

import random
from collections import OrderedDict

############## config 

## search range
range_dict = OrderedDict()
range_dict["emb_sz"] = [128,256]
range_dict["enc_hidden_size"] = [256,512]
range_dict["enc_num_layers"] = [1,2,3]
range_dict["dec_hid_sz"] = [256,512]
range_dict["dec_n_layer"] = [1,2]
range_dict["ladder_d_size"] = ([128,64,32],[256,128,64])
range_dict["ladder_z_size"] = ([16,8,4],[32,16,8])
range_dict["ladder_z2z_layer_size"] = ([8,16],[16,32])
range_dict["dropout"] = [0.1,0.2,0.3]

n_combo = 100

############## random combination

hyper_rand = []
for _ in range(n_combo):
    choice = []
    for k,v in range_dict.items():
        choice.append(random.choice(v))
    hyper_rand.append(choice)
    
for h in hyper_rand:
    print(h)

############## save

with open("hyper_rand.csv","w") as fo:
    for i,c in enumerate(hyper_rand):
        c = [f"combo_{i}"]+c
        for e in c[:-1]:
            if isinstance(e, list):
                e = " ".join(list(map(str,e)))
            fo.write(f"{e},")
        fo.write(f"{c[-1]}\n")

            

