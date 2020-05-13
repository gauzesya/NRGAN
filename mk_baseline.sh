#!/bin/sh

python make_base_wav.py \
  --n_test_data 100 \
  --noised_label_tr data/label/train_noised \
  --denoise_label_tr data/label/train_denoise \
  --noised_label_te data/label/test_noised \
  --denoise_label_te data/label/test_denoise \
  --test_out_dir wav/baseline
