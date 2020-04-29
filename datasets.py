# -*- coding: utf-8 -*-

'''
Dataset for NRGAN
'''

import os
from scipy.signal import lfilter
import torch
import soundfile as sf
import numpy as np
from utils import read_label


class NoisedAndDenoiseAudioDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            noised_label,
            denoise_label=None,
            n_sample=16384,
            n_overlap=8192,
            fs=16000
            ):

        self._n_sample = n_sample
        self._n_overlap = n_overlap
        self._fs = 16000

        self._noised_data = self._make_data(noised_label)
        self._noised_chunks = self._extract_chunks(
                self._noised_data
                )
        self._noised_chunks_len = self._noised_chunks.shape[0]

        if denoise_label is not None:
            self._denoise_data = self._make_data(denoise_label)
            self._denoise_chunks = self._extract_chunks(
                    self._denoise_data
                    )
            self._denoise_chunks_len = self._denoise_chunks.shape[0]
            self._chunks_len = max(self._noised_chunks_len, self._denoise_chunks_len)
            self._noised_index = np.arange(self._chunks_len) % self._noised_chunks_len
            self._denoise_index = np.arange(self._chunks_len) % self._denoise_chunks_len
        else:
            self._denoise_data = None
            self._denoise_chunks = None
            self._denoise_chunks_len = None
            self._chunks_len = self._noised_chunks_len
            self._noised_index = np.arange(self._chunks_len)
            self._denoise_index = None



    def shuffle_data(self):
        self._noised_index = np.random.permutation(self._chunks_len) % self._noised_chunks_len
        if self._denoise_data is not None:
            self._denoise_index = np.random.permutation(self._chunks_len) % self._denoise_chunks_len


    def get_test_data_by_file(self, n_data=5, data_type='noised', is_shuffle=False):
        assert(data_type=='noised' or data_type=='denoise')

        if data_type=='noised':
            data = self._noised_data
        elif data_type=='denoise':
            data = self._denoise_data

        if is_shuffle is True:
            data = np.random.permutation(data)

        data = data[:n_data]

        test_data = []
        for d in data:
            wav = d['wav']
            chunks_by_wav = self._extract_chunks_from_wav(
                    wav,
                    self._n_sample,
                    0)
            d['chunks'] = chunks_by_wav
            test_data.append(d)

        return test_data


    def save_denoised_wav(self, denoise_model, output_dir, n_data=5, is_shuffle=False, decostr=''):

        # denoise_model is function that recieves `chuncks' and returns numpy array of the same size
        os.makedirs(output_dir, exist_ok=True)
        test_data = self.get_test_data_by_file(
                n_data=n_data,
                data_type='noised',
                is_shuffle=is_shuffle)

        for td in test_data:
            name = td['name']
            wav_len = td['len']
            chunks = td['chunks']

            denoised_chunks = denoise_model(chunks)

            denoised = []
            for dc in denoised_chunks:
                denoised = denoised + list(dc)
            denoised = np.array(denoised[:wav_len])
            denoised = self._de_emphasis(denoised) # deemphasis

            output_path = os.path.join(output_dir, decostr+name)
            sf.write(output_path, denoised, self._fs)
            


    def _make_data(self, label):

        paths = read_label(label)
        data = []
        for fp in paths:
            basename = os.path.basename(fp)
            wav, fs = sf.read(fp)
            assert(self._fs==fs)
            wav = self._pre_emphasis(wav)
            wav_length = wav.shape[0]

            dd = {
                    'name': basename,
                    'len': wav_length,
                    'wav': wav
                    }
            data.append(dd)

        return data


    def _extract_chunks(self, data):
        chunks = []

        for d in data:
            wav = d['wav']
            chunks_by_wav = self._extract_chunks_from_wav(
                    wav,
                    self._n_sample,
                    self._n_overlap)
            chunks = chunks + list(chunks_by_wav)

        chunks = np.array(chunks)
        return chunks


    def _extract_chunks_from_wav(self, wav, n_sample, n_overlap):

        n_step = n_sample - n_overlap
        chunks = []
        wav_len = wav.shape[0]
        n_chunk = int(np.ceil(wav_len / n_step))

        for i in range(n_chunk):
            start = i * n_step
            if start+n_sample > wav_len:
                chunk = np.pad(wav[start:start+n_sample], [0, (start+n_sample)-wav_len], 'constant')
            else:
                chunk = wav[start:start+n_sample]
            chunks.append(chunk)

        chunks = np.array(chunks)
        return chunks


    def _pre_emphasis(self, wave, coef=0.95):
        return lfilter([1.0, -coef], [1], wave)


    def _de_emphasis(self, wave, coef=0.95):
        return lfilter([1], [1.0, -coef], wave)


    def __len__(self):
        return self._chunks_len


    def __getitem__(self, index):
        if self._denoise_data is not None:
            items = {
                    'noised': self._noised_chunks[self._noised_index][index],
                    'denoise': self._denoise_chunks[self._denoise_index][index]
                }
        else:
            items = {
                    'noised': self._noised_chunks[self._noised_index][index]
                }
        return items
