#!/bin/bash
while IFS="," read f1 f2 f3 f4 f5 f6 f7 f8 f9 f10
do
        TGT_DIR=../tmp/exp.hyp_sea/$f1
        mkdir -p $TGT_DIR
        echo Doing $f1
        python train.py --device cuda:2 --model_save $TGT_DIR/model.pt --log_path $TGT_DIR/log.csv \
                --emb_sz $f2 \
                --enc_hidden_size $f3 \
                --enc_num_layers $f4 \
                --dec_hid_sz $f5 \
                --dec_n_layer $f6 \
                --ladder_d_size $f7 \
                --ladder_z_size $f8 \
                --ladder_z2z_layer_size $f9 \
                --dropout $f10 \
                --lr_anr_type const \
                --n_epoch 1
done < hyper_rand.csv
