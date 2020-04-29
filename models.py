# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DownSample(nn.Module):

    def __init__(self, n_in, n_out, relu='prelu', batchnorm=True):
        super(DownSample, self).__init__()

        assert(relu in ['relu', 'lrelu', 'prelu'])
        self.conv = nn.Conv1d(n_in, n_out, kernel_size=32, stride=2, padding=15)
        if relu=='relu':
            self.activate_func = nn.ReLU()
        elif relu=='lrelu':
            self.activate_func = nn.LeakyReLU()
        elif relu=='prelu':
            self.activate_func = nn.PReLU()

        if batchnorm is True:
            self.bn = nn.BatchNorm1d(n_out)
        else:
            self.bn = None


    def forward(self, x):
        x = x.float()
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activate_func(x)

        return x


class UpSample(nn.Module):

    def __init__(self, n_in, n_out, relu='prelu', batchnorm=True):
        super(UpSample, self).__init__()

        assert(relu in ['relu', 'lrelu', 'prelu'])
        self.conv = nn.ConvTranspose1d(n_in, n_out, kernel_size=32, stride=2, padding=15)
        if relu=='relu':
            self.activate_func = nn.ReLU()
        elif relu=='lrelu':
            self.activate_func = nn.LeakyReLU()
        elif relu=='prelu':
            self.activate_func = nn.PReLU()

        if batchnorm is True:
            self.bn = nn.BatchNorm1d(n_out)
        else:
            self.bn = None


    def forward(self, x):
        x = x.float()
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activate_func(x)

        return x


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.ds1 = DownSample(1, 16)
        self.ds2 = DownSample(16, 32)
        self.ds3 = DownSample(32, 32)
        self.ds4 = DownSample(32, 64)
        self.ds5 = DownSample(64, 64)
        self.ds6 = DownSample(64, 128)
        self.ds7 = DownSample(128, 128)
        self.ds8 = DownSample(128, 256)
        self.ds9 = DownSample(256, 256)
        self.ds10 = DownSample(256, 512)
        self.ds11 = DownSample(512, 1024)

        self.dropout = nn.Dropout(p=0.5)

        self.us1 = UpSample(1024, 512)
        self.us2 = UpSample(1024, 256)
        self.us3 = UpSample(512, 256)
        self.us4 = UpSample(512, 128)
        self.us5 = UpSample(256, 128)
        self.us6 = UpSample(256, 64)
        self.us7 = UpSample(128, 64)
        self.us8 = UpSample(128, 32)
        self.us9 = UpSample(64, 32)
        self.us10 = UpSample(64, 16)
        self.us11 = UpSample(32, 1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.float()
        x = torch.unsqueeze(x, 1)

        # down-sampling
        sc1 = self.ds1(x)
        sc2 = self.ds2(sc1)
        sc3 = self.ds3(sc2)
        sc4 = self.ds4(sc3)
        sc5 = self.ds5(sc4)
        sc6 = self.ds6(sc5)
        sc7 = self.ds7(sc6)
        sc8 = self.ds8(sc7)
        sc9 = self.ds9(sc8)
        sc10 = self.ds10(sc9)
        sc11 = self.ds11(sc10)

        x = self.dropout(sc11)

        # up-sampling
        x = self.us1(x)
        x = torch.cat((x, sc10), dim=1)
        x = self.us2(x)
        x = torch.cat((x, sc9), dim=1)
        x = self.us3(x)
        x = torch.cat((x, sc8), dim=1)
        x = self.us4(x)
        x = torch.cat((x, sc7), dim=1)
        x = self.us5(x)
        x = torch.cat((x, sc6), dim=1)
        x = self.us6(x)
        x = torch.cat((x, sc5), dim=1)
        x = self.us7(x)
        x = torch.cat((x, sc4), dim=1)
        x = self.us8(x)
        x = torch.cat((x, sc3), dim=1)
        x = self.us9(x)
        x = torch.cat((x, sc2), dim=1)
        x = self.us10(x)
        x = torch.cat((x, sc1), dim=1)
        x = self.us11(x)

        x = self.tanh(x)
        x = torch.squeeze(x, dim=1)

        return x


class Discriminator(nn.Module):

    def __init__(self, n_sample):
        super(Discriminator, self).__init__()

        self.ds1 = DownSample(1, 16, relu='lrelu')
        self.ds2 = DownSample(16, 32, relu='lrelu')
        self.ds3 = DownSample(32, 32, relu='lrelu')
        self.ds4 = DownSample(32, 64, relu='lrelu')
        self.ds5 = DownSample(64, 64, relu='lrelu')
        self.ds6 = DownSample(64, 128, relu='lrelu')
        self.ds7 = DownSample(128, 128, relu='lrelu')
        self.ds8 = DownSample(128, 256, relu='lrelu')
        self.ds9 = DownSample(256, 256, relu='lrelu')
        self.ds10 = DownSample(256, 512, relu='lrelu')
        self.ds11 = DownSample(512, 1024, relu='lrelu')

        self.flatten = Flatten()
        self.fc = nn.Linear(1024*int(n_sample/(2**11)), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        x = torch.unsqueeze(x, 1)

        # down-sampling
        x = self.ds1(x)
        x = self.ds2(x)
        x = self.ds3(x)
        x = self.ds4(x)
        x = self.ds5(x)
        x = self.ds6(x)
        x = self.ds7(x)
        x = self.ds8(x)
        x = self.ds9(x)
        x = self.ds10(x)
        x = self.ds11(x)

        x = self.flatten(x)
        x = self.fc(x)

        x = self.sigmoid(x)

        return x


if __name__=='__main__':
    from datasets import NoisedAndDenoiseAudioDataset
    from torch.utils.data import DataLoader
    train_dataset = NoisedAndDenoiseAudioDataset(
            'data/label/train_noised',
            'data/label/train_denoise'
            )

    dl = DataLoader(train_dataset, batch_size=1, num_workers=2, shuffle=True)

    gen = Generator()
    dis = Discriminator(16384)
    for d in dl:
        noised = d['noised']
        denoise = d['denoise']

        converted = gen(noised)
        a = dis(noised)
        print(a)
        #print(converted)
        quit()

