<<COMMENT
usage: main.py [-h] [--n_epochs N_EPOCHS] [--batchsize BATCHSIZE]
               [--lr_g LR_G] [--lr_d LR_D] [--l1_ratio L1_RATIO] [--only_test]
               [--test_epoch TEST_EPOCH] [--n_test_data N_TEST_DATA]
               [--is_shuffle_test] [--n_sample N_SAMPLE]
               [--n_overlap N_OVERLAP] [--noised_label_tr NOISED_LABEL_TR]
               [--denoise_label_tr DENOISE_LABEL_TR]
               [--noised_label_te NOISED_LABEL_TE]
               [--denoise_label_te DENOISE_LABEL_TE] [--exp_dir EXP_DIR]
               [--test_dir TEST_DIR] [--save_interval SAVE_INTERVAL]

NRGAN

optional arguments:
  -h, --help            show this help message and exit
  --n_epochs N_EPOCHS   The number of training epochs
  --batchsize BATCHSIZE
                        The size of batch
  --lr_g LR_G           Learning rate of generator
  --lr_d LR_D           Learning rate of discriminator
  --l1_ratio L1_RATIO   Ratio of L1 norm for generator training
  --only_test           Can be used for only denoising from a trained model
  --test_epoch TEST_EPOCH
                        Only testing: The model with the epoch is chosen for
                        testing (-1 means the latest)
  --n_test_data N_TEST_DATA
                        The number of wave file generated in the test
  --is_shuffle_test     Whether data is shuffled during testing
  --n_sample N_SAMPLE   The number of sample
  --n_overlap N_OVERLAP
                        The overlap number
  --noised_label_tr NOISED_LABEL_TR
                        The label path of noised data for training
  --denoise_label_tr DENOISE_LABEL_TR
                        The label path of denoised data for training
  --noised_label_te NOISED_LABEL_TE
                        The label path of noised data for testing
  --denoise_label_te DENOISE_LABEL_TE
                        The label path of denoised data for testing
  --exp_dir EXP_DIR     The directry path that contains experiments (or empty
                        dir)
  --test_dir TEST_DIR   Only testing: The directory of denoised wav
  --save_interval SAVE_INTERVAL
                        The interval of epoch for saving and testing
COMMENT

python main.py \
  --n_epochs 50 \
  --lr_g 1e-3 \
  --lr_d 1e-4 \
  --l1_ratio 1 \
  --noised_label_tr data/label/train_noised \
  --denoise_label_tr data/label/train_denoise \
  --noised_label_te data/label/test_noised \
  --exp_dir exp \
  --save_interval 10

<<TEST
python main.py \
  --only_test \
  --is_shuffle_test \
  --noised_label_te data/label/test_noised_tiny \
  --exp_dir exp \
  --test_dir wav/test
TEST
