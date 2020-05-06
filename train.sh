#!/bin/sh

<<TRAIN
python main.py \
  --n_epochs 50 \
  --batchsize 4 \
  --opt_g adam \
  --opt_d adam \
  --lr_g 1e-3 \
  --lr_d 1e-4 \
  --l1_ratio 1 \
  --noised_label_tr data/label/train_noised \
  --denoise_label_tr data/label/train_denoise \
  --noised_label_te data/label/test_noised \
  --exp_dir exp \
  --out_dir exp \
  --save_interval 5 \
  --is_pair
TRAIN

<<TRAIN_TINY
python main.py \
  --n_epochs 5 \
  --batchsize 4 \
  --opt_g adam \
  --opt_d adam \
  --lr_g 1e-3 \
  --lr_d 1e-4 \
  --l1_ratio 1 \
  --noised_label_tr data/label/train_noised_tiny \
  --denoise_label_tr data/label/train_denoise_tiny \
  --noised_label_te data/label/test_noised_tiny \
  --exp_dir exp \
  --out_dir exp \
  --save_interval 1 \
  --is_pair
TRAIN_TINY

<<RE_TRAIN
python main.py \
  --n_epochs 50 \
  --batchsize 4 \
  --lr_g 1e-3 \
  --lr_d 1e-4 \
  --l1_ratio 1 \
  --noised_label_tr data/label/train_noised \
  --denoise_label_tr data/label/train_denoise \
  --noised_label_te data/label/test_noised \
  --load_epochs 5 \
  --exp_dir exp \
  --out_dir exp2 \
  --save_interval 5 \
  --is_pair
RE_TRAIN
