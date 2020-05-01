#!/bin/sh

<<TRAIN
python main.py \
  --n_epochs 50 \
  --batchsize 4 \
  --lr_g 1e-3 \
  --lr_d 1e-4 \
  --l1_ratio 1 \
  --dropout_prob_g 0.1 \
  --dropout_prob_d 0.5 \
  --noised_label_tr data/label/train_noised \
  --denoise_label_tr data/label/train_denoise \
  --noised_label_te data/label/test_noised \
  --exp_dir exp \
  --out_dir exp \
  --save_interval 5 \
  --is_pair
TRAIN

<<RE_TRAIN
python main.py \
  --n_epochs 50 \
  --batchsize 4 \
  --lr_g 1e-3 \
  --lr_d 1e-4 \
  --l1_ratio 1 \
  --dropout_prob_g 0.1 \
  --dropout_prob_d 0.5 \
  --noised_label_tr data/label/train_noised \
  --denoise_label_tr data/label/train_denoise \
  --noised_label_te data/label/test_noised \
  --load_epochs 5 \
  --exp_dir exp \
  --out_dir exp2 \
  --save_interval 5 \
  --is_pair
RE_TRAIN

<<TEST
python main.py \
  --only_test \
  --is_shuffle_test \
  --noised_label_te data/label/test_noised \
  --load_epochs 5 \
  --exp_dir exp \
  --test_out_dir test
TEST
