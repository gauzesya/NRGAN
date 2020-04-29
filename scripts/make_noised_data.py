# -*- coding: utf-8 -*-

'''
Make noised waveform from simple normal distribution
'''

import os
import glob
import argparse
import numpy as np
import soundfile as sf

def main(conf):

    os.makedirs(conf['out_dir'], exist_ok=True)
    wav_file_paths = glob.glob(os.path.join(conf['wave_dir'], '*.wav'))

    for file_path in wav_file_paths:
        basename = os.path.basename(file_path)
        save_path = os.path.join(conf['out_dir'], basename)

        data, fs = sf.read(file_path)
        wav_length = data.shape[0]

        print('making {}...'.format(save_path))
        noised = data + np.random.normal(0 , conf['std'], wav_length)

        sf.write(save_path, noised, fs)




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Make noised waveform')
    parser.add_argument('--std', '-s', type=float, default=0.01,
            help='Standard deviation of normal distribution')
    parser.add_argument('--wave_dir', '-w', type=str,
            help='The directry path which contains original waveform')
    parser.add_argument('--out_dir', '-o', type=str,
            help='The directry path to output a training result')
    args = parser.parse_args()
    conf = vars(args)

    main(conf)
