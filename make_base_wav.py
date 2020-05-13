# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
import numpy as np
import glob
import os
import json

from pyutils.progressbar import progressbar
from pyutils.logger import logger

from models import Generator, Discriminator
from datasets import NoisedAndDenoiseAudioDataset


def test(conf):


    noised_label_tr = conf['noised_label_tr']
    denoise_label_tr = conf['denoise_label_tr']
    noised_label_te = conf['noised_label_te']
    denoise_label_te = conf['denoise_label_te']

    n_test_data = conf['n_test_data']
    is_shuffle_test = conf['is_shuffle_test']

    test_out_dir = conf['test_out_dir']

    n_sample = conf['n_sample']
    n_overlap = conf['n_overlap']

    # load dataset
    train_dataset = NoisedAndDenoiseAudioDataset(
            noised_label_tr,
            denoise_label_tr,
            n_sample=n_sample,
            n_overlap=n_overlap
            )
    norm_param = train_dataset.get_norm_param()
    test_dataset = NoisedAndDenoiseAudioDataset(
            noised_label_te,
            denoise_label_te,
            n_sample=n_sample,
            n_overlap=n_overlap
            )

    # device (cpu or cuda)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    denoised_dir = test_out_dir
    os.makedirs(denoised_dir)

    # generate test waveform
    test_dataset.save_baseline_wav(
            denoised_dir,
            n_data = n_test_data,
            is_shuffle = is_shuffle_test,
            norm_param = norm_param
            )


def main(conf):

    test(conf)



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Make baseline')
    # For training and testing 
    parser.add_argument('--n_test_data', type=int, default=5,
            help='The number of wave file generated in the test')
    parser.add_argument('--is_shuffle_test', action='store_true',
            help='Whether data is shuffled during testing')

    # For dataset 
    parser.add_argument('--n_sample', type=int, default=16384,
            help='The number of sample')
    parser.add_argument('--n_overlap', type=int, default=8192,
            help='The overlap number')
    parser.add_argument('--noised_label_tr', type=str,
            help='The label path of noised data for training')
    parser.add_argument('--denoise_label_tr', type=str,
            help='The label path of denoised data for training')
    parser.add_argument('--noised_label_te', type=str, default=None,
            help='The label path of noised data for testing')
    parser.add_argument('--denoise_label_te', type=str,
            help='The label path of denoised data for testing')

    # For experiments
    parser.add_argument('--test_out_dir', type=str, default='testdir',
            help='The directory path for saving wav')
    args = parser.parse_args()
    conf = vars(args)

    main(conf)
