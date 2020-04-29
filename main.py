# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
import numpy as np
import os
import json

from pyutils.progressbar import progressbar
from pyutils.logger import logger

from models import Generator, Discriminator
from datasets import NoisedAndDenoiseAudioDataset


def make_or_load_experiment(conf):

    # set outputs dir and load pre-settings (if exist)
    only_test = conf['only_test']
    n_epochs = conf['n_epochs']
    test_epoch = conf['test_epoch']
    exp_dir = conf['exp_dir']
    save_interval = conf['save_interval']

    model_dir = None
    pre_results = None
    saved_epochs = 0

    
    if only_test is True:
        if test_epoch==-1:
            # load pre-results
            with open(os.path.join(exp_dir, 'result.csv'), 'r') as f:
                pre_results = f.readlines()
            pre_num_epochs = len(pre_results) - 1

            # load pre-config
            with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
                n_conf = json.load(f)
            pre_interval = n_conf['save_interval']
            if pre_num_epochs == n_conf['n_epochs']:
                saved_epochs = pre_num_epochs
            else:
                saved_epochs = pre_num_epochs - (pre_num_epochs%pre_interval)
            model_dir = os.path.join(exp_dir, 'epoch_{}'.format(saved_epochs))
        else:
            saved_epochs = test_epochs
            model_dir = os.path.join(exp_dir, 'epoch_{}'.format(test_epochs))

    else:

        try:
            os.makedirs(exp_dir)
        except:
            assert os.path.exists(os.path.join(exp_dir, 'result.csv')), '{} not have result.csv'.format(exp_dir)
            # load pre-results
            with open(os.path.join(exp_dir, 'result.csv'), 'r') as f:
                pre_results = f.readlines()
            pre_num_epochs = len(pre_results) - 1
            assert(pre_num_epochs < n_epochs)

            # load pre-config
            assert os.path.exists(os.path.join(exp_dir, 'config.json')), '{} not have config.json'.format(exp_dir)
            with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
                conf = json.load(f)
            pre_interval = conf['save_interval']
            if pre_num_epochs == conf['n_epochs']:
                saved_epochs = pre_num_epochs
            else:
                saved_epochs = pre_num_epochs - (pre_num_epochs%pre_interval)
            pre_n_epochs = saved_epochs + 1
            model_dir = os.path.join(exp_dir, 'epoch_{}'.format(saved_epochs))
            results = pre_results[1:saved_epochs+1]

            # update config
            conf['n_epochs'] = n_epochs
            conf['save_interval'] = save_interval

            # log
            print('data re-loaded from {}'.format(model_dir))
            print('re-start training from {} epoch'.format(pre_n_epochs))

    return conf, model_dir, saved_epochs, pre_results


def model(noised, netG, device):

    noised = torch.from_numpy(noised.astype(np.float32)).clone()
    noised = noised.to(device)
    converted = netG(noised)
    converted = converted.to('cpu').detach().numpy()

    return converted


def train(conf):

    # parameters
    n_epochs = conf['n_epochs']
    batchsize = conf['batchsize']
    lr_g = conf['lr_g']
    lr_d = conf['lr_d']
    l1_ratio = conf['l1_ratio']
    save_interval = conf['save_interval']
    n_sample = conf['n_sample']
    n_overlap = conf['n_overlap']

    noised_label_tr = conf['noised_label_tr']
    denoise_label_tr = conf['denoise_label_tr']
    noised_label_te = conf['noised_label_te']

    exp_dir = conf['exp_dir']

    if noised_label_te is not None:
        is_testing = True
    else:
        is_testing = False

    # load experiment (or make new experiment)
    conf, model_dir, saved_epochs, pre_results = make_or_load_experiment(conf)
    pre_n_epochs = saved_epochs + 1


    # load dataset
    train_dataset = NoisedAndDenoiseAudioDataset(
            noised_label_tr,
            denoise_label_tr
            )
    data_num = len(train_dataset)
    if is_testing:
        test_dataset = NoisedAndDenoiseAudioDataset(
                noised_label_te,
                None
                )
    # data shuffle for making noised and denoise data unpair
    train_dataset.shuffle_data()
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batchsize, num_workers=4, shuffle=True)

    # device (cpu or cuda)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    # set model
    netG = Generator()
    netG = netG.to(device)
    netD = Discriminator(n_sample)
    netD = netD.to(device)

    # create optimiser
    optG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.9))

    # load models (if exists)
    if model_dir is not None:
        netG.load_state_dict(torch.load(os.path.join(model_dir, "netG.pt")))
        optG.load_state_dict(torch.load(os.path.join(model_dir, "optG.pt")))
        netD.load_state_dict(torch.load(os.path.join(model_dir, "netD.pt")))
        optD.load_state_dict(torch.load(os.path.join(model_dir, "optD.pt")))

    # save config
    conf_json = json.dumps(conf, indent=2)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        f.write(conf_json)

    # logger
    with logger(os.path.join(exp_dir, 'result.csv')) as printl:
        printl('epoch, loss_D, loss_G, loss_L1', only_file=True)
        if pre_results is not None:
            for pr in pre_results:
                printl(pr[:-1], only_file=True)

        for epoch in range(pre_n_epochs, n_epochs+1):

            total_loss_D = 0.
            total_loss_G = 0.
            total_loss_L1 = 0.

            for dl in progressbar(
                    train_loader, decostr='epoch {:03d}'.format(epoch)):

                # set data
                noised = dl['noised']
                noised = noised.to(device)
                denoise = dl['denoise']
                denoise = denoise.to(device)
                bs = noised.shape[0]

                # Discriminator training
                loss_D = 0.
                fake_D = netD(netG(noised))
                real_D = netD(denoise)
                loss_D += (fake_D**2).mean()
                loss_D += ((real_D-1)**2).mean()
                total_loss_D += loss_D.to('cpu').detach().numpy() * bs
                netD.zero_grad()
                loss_D.backward()
                optD.step()

                # Generator training
                fake = netG(noised)
                fake_D = netD(fake)
                loss_G = ((fake_D-1)**2).mean()
                total_loss_G += loss_G.to('cpu').detach().numpy() * bs
                loss_L1 = F.l1_loss(noised, fake).mean()
                total_loss_L1 += loss_L1.to('cpu').detach().numpy() * bs
                loss_L1 = l1_ratio * loss_L1

                netG.zero_grad()
                (loss_G + loss_L1).backward()
                optG.step()

            # compute mean loss nad print
            total_loss_D /= data_num
            total_loss_G /= data_num
            total_loss_L1 /= data_num

            print('epoch {0:03d},  loss_D: {1:.5f},  loss_G: {2:.05f},  loss_L1: {3:.05f}'
                    .format(epoch, total_loss_D, total_loss_G, total_loss_L1))
            printl('{0:d}, {1:.5f}, {2:.05f}, {3:.05f}'
                    .format(epoch, total_loss_D, total_loss_G, total_loss_L1), only_file=True)


            if epoch%save_interval==0 or epoch==n_epochs:

                # save model
                save_dir = os.path.join(exp_dir, 'epoch_{}'.format(epoch))
                os.makedirs(save_dir)
                torch.save(netG.state_dict(), os.path.join(save_dir, "netG.pt"))
                torch.save(optG.state_dict(), os.path.join(save_dir, "optG.pt"))
                torch.save(netD.state_dict(), os.path.join(save_dir, "netD.pt"))
                torch.save(optD.state_dict(), os.path.join(save_dir, "optD.pt"))

                if is_testing:
                    denoised_dir = os.path.join(save_dir, 'denoised')
                    os.makedirs(denoised_dir)

                    test_dataset.save_denoised_wav(
                            lambda x: model(x, netG, device),
                            denoised_dir,
                            n_data=conf['n_test_data'],
                            is_shuffle=conf['is_shuffle_test']
                            )

def test(conf):

    n_sample = conf['n_sample']
    n_overlap = conf['n_overlap']

    noised_label_te = conf['noised_label_te']

    # load experiment
    conf, model_dir, saved_epochs, pre_results = make_or_load_experiment(conf)
    assert(model_dir is not None)

    # load dataset
    test_dataset = NoisedAndDenoiseAudioDataset(
            noised_label_te,
            None
            )

    # device (cpu or cuda)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    # set model
    netG = Generator()
    netG = netG.to(device)
    netD = Discriminator(n_sample)
    netD = netD.to(device)

    # load models 
    netG.load_state_dict(torch.load(os.path.join(model_dir, "netG.pt")))
    netD.load_state_dict(torch.load(os.path.join(model_dir, "netD.pt")))

    denoised_dir = conf['test_dir']
    os.makedirs(denoised_dir)

    test_dataset.save_denoised_wav(
            lambda x: model(x, netG, device),
            denoised_dir,
            n_data=conf['n_test_data'],
            is_shuffle=conf['is_shuffle_test']
            )


def main(conf):

    if conf['only_test'] is False:
        train(conf)
    elif conf['noised_label_te'] is not None:
        test(conf)
    else:
        print('Test noised label is needed!')



if __name__=='__main__':

    parser = argparse.ArgumentParser(description='NRGAN')
    # For training and testing 
    parser.add_argument('--n_epochs', type=int, default=50,
            help='The number of training epochs')
    parser.add_argument('--batchsize', type=int, default=32,
            help='The size of batch')
    parser.add_argument('--lr_g', type=float, default=1e-4,
            help='Learning rate of generator')
    parser.add_argument('--lr_d', type=float, default=1e-5,
            help='Learning rate of discriminator')
    parser.add_argument('--l1_ratio', type=float, default=1,
            help='Ratio of L1 norm for generator training')
    parser.add_argument('--only_test', action='store_true',
            help='Can be used for only denoising from a trained model')
    parser.add_argument('--test_epoch', type=int, default=-1,
            help='Only testing: The model with the epoch is chosen for testing (-1 means the latest)')
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

    # For experiments
    parser.add_argument('--exp_dir', type=str, default='exp',
            help='The directry path that contains experiments (or empty dir)')
    parser.add_argument('--test_dir', type=str, default='testdir',
            help='Only testing: The directory of denoised wav')
    parser.add_argument('--save_interval', type=int, default=10,
            help='The interval of epoch for saving and testing')
    args = parser.parse_args()
    conf = vars(args)

    main(conf)
