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
            fs=16000,
            is_pair=False
            ):

        self._noised_label = noised_label
        self._denoise_label = denoise_label
        self._n_sample = n_sample
        self._n_overlap = n_overlap
        self._fs = fs
        self._is_pair = is_pair

        noised_data, noised_max, noised_min = self._make_data(self._noised_label)
        self._noised_chunks = self._extract_chunks(
                noised_data,
                noised_max,
                noised_min
                )
        self._noised_chunks_len = self._noised_chunks.shape[0]

        if denoise_label is not None:
            denoise_data, denoise_max, denoise_min = self._make_data(self._denoise_label)
            self._denoise_chunks = self._extract_chunks(
                    denoise_data,
                    denoise_max,
                    denoise_min
                    )

            if self._is_pair: # pair check
                assert len(noised_data)==len(denoise_data)
                for noised, denoise in zip(noised_data, denoise_data):
                    assert noised['len']==denoise['len']
                print('pair check passed')

                self._denoise_chunks_len = self._noised_chunks_len
                self._chunks_len = self._noised_chunks_len
                self._noised_index = np.arange(self._chunks_len)
                self._denoise_index = np.arange(self._chunks_len)

            else:
                self._denoise_chunks_len = self._denoise_chunks.shape[0]
                self._chunks_len = max(self._noised_chunks_len, self._denoise_chunks_len)
                self._noised_index = np.arange(self._chunks_len) % self._noised_chunks_len
                self._denoise_index = np.arange(self._chunks_len) % self._denoise_chunks_len

        else:
            denoise_max = 0.0
            denoise_min = 0.0
            self._denoise_chunks = None
            self._denoise_chunks_len = None
            self._chunks_len = self._noised_chunks_len
            self._noised_index = np.arange(self._chunks_len)
            self._denoise_index = None

        self._norm_param = np.array([noised_max, noised_min, denoise_max, denoise_min])
            



    def shuffle_data(self):
        self._noised_index = np.random.permutation(self._chunks_len) % self._noised_chunks_len
        if self._denoise_chunks is not None:
            if self._is_pair:
                self._denoise_index = self._noised_index
            else:
                self._denoise_index = np.random.permutation(self._chunks_len) % self._denoise_chunks_len


    def get_norm_param(self):
        return self._norm_param


    def get_test_data_by_file(self, n_data=5, data_type='noised', is_shuffle=False):
        assert(data_type=='noised' or data_type=='denoise')

        if data_type=='noised':
            data, _, _ = self._make_data(self._noised_label)
        elif data_type=='denoise':
            data, _, _ = self._make_data(self._denoise_label)

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


    def save_denoised_wav(self, denoise_model, output_dir, n_data=5, is_shuffle=False, decostr='', norm_param=None):

        # denoise_model is function that recieves `chuncks' and returns numpy array of the same size
        os.makedirs(output_dir, exist_ok=True)
        test_data = self.get_test_data_by_file(
                n_data=n_data,
                data_type='noised',
                is_shuffle=is_shuffle)
        if norm_param is not None:
            _, _, denoise_max, denoise_min = norm_param
        else:
            _, _, denoise_max, denoise_min = self._norm_param

        for td in test_data:
            name = td['name']
            wav_len = td['len']
            chunks = td['chunks']

            denoised_chunks = denoise_model(chunks)

            denoised = []
            for dc in denoised_chunks:
                denoised = denoised + list(dc)
            denoised = np.array(denoised[:wav_len])
            denoised = self._de_norm(denoised, denoise_max, denoise_min) # denorm
            denoised = self._de_emphasis(denoised) # deemphasis

            output_path = os.path.join(output_dir, decostr+name)
            sf.write(output_path, denoised, self._fs)
            

    # function for generationg baseline waveform
    def save_baseline_wav(self, output_dir, n_data=5, is_shuffle=False, decostr='', norm_param=None):

        # denoise_model is function that recieves `chuncks' and returns numpy array of the same size
        os.makedirs(output_dir, exist_ok=True)
        test_data = self.get_test_data_by_file(
                n_data=n_data,
                data_type='denoise',
                is_shuffle=is_shuffle)
        if norm_param is not None:
            _, _, denoise_max, denoise_min = norm_param
        else:
            _, _, denoise_max, denoise_min = self._norm_param

        for td in test_data:
            name = td['name']
            wav_len = td['len']
            chunks = td['chunks']

            denoised_chunks = chunks

            denoised = []
            for dc in denoised_chunks:
                denoised = denoised + list(dc)
            denoised = np.array(denoised[:wav_len])
            denoised = self._de_norm(denoised, denoise_max, denoise_min) # denorm
            denoised = self._de_emphasis(denoised) # deemphasis

            output_path = os.path.join(output_dir, decostr+name)
            sf.write(output_path, denoised, self._fs)


    def _make_data(self, label):

        paths = read_label(label)
        data = []
        n_max = -np.inf
        n_min = np.inf
        for fp in paths:
            basename = os.path.basename(fp)
            wav, fs = sf.read(fp)
            assert(self._fs==fs)
            wav = self._pre_emphasis(wav)
            wav_length = wav.shape[0]
            n_max = np.max([n_max, np.max(wav)])
            n_min = np.min([n_min, np.min(wav)])

            dd = {
                    'name': basename,
                    'len': wav_length,
                    'wav': wav
                    }
            data.append(dd)

        return data, n_max, n_min

    def _norm(self, chunks, n_max, n_min):
        sa = (n_max + n_min) / 2
        ma = (n_max - n_min) / 2
        chunks = (chunks - sa) / ma * 0.99
        return chunks

    def _de_norm(self, chunks, n_max, n_min):
        sa = (n_max + n_min) / 2
        ma = (n_max - n_min) / 2
        chunks = (chunks * ma / 0.99) + sa
        return chunks

    def _extract_chunks(self, data, n_max, n_min):
        chunks = []

        for d in data:
            wav = d['wav']
            chunks_by_wav = self._extract_chunks_from_wav(
                    wav,
                    self._n_sample,
                    self._n_overlap)
            chunks = chunks + list(chunks_by_wav)

        chunks = np.array(chunks)
        chunks = self._norm(chunks, n_max, n_min) # normalise in [-0.99, 0.99]
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
        if self._denoise_chunks is not None:
            items = {
                    'noised': self._noised_chunks[self._noised_index][index],
                    'denoise': self._denoise_chunks[self._denoise_index][index]
                }
        else:
            items = {
                    'noised': self._noised_chunks[self._noised_index][index]
                }
        return items
