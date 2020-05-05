#!/bin/sh

<<TEST
python main.py \
  --only_test \
  --is_shuffle_test \
  --noised_label_te data/label/test_noised \
  --load_epochs 5 \
  --exp_dir exp \
  --test_out_dir test
TEST
