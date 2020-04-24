# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： processor_tutorial.py
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/4/22  23:17
-------------------------------------------------
   Change Activity:
          2020/4/22 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import madmom
import numpy as np
import matplotlib.pyplot as plt

sig = madmom.audio.signal.Signal('data/sample.wav')
fs = madmom.audio.signal.FramedSignal(sig)
print('fs',fs,type(fs))
stft = madmom.audio.stft.ShortTimeFourierTransform(fs)
print('stft',stft,type(stft))
spec = madmom.audio.spectrogram.Spectrogram(stft)
print('spec',spec,type(spec))
filt = madmom.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24)
print('filt',filt,type(filt))
log = madmom.audio.spectrogram.LogarithmicSpectrogram(filt)
print('log',log,type(log))
diff = madmom.audio.spectrogram.SpectrogramDifference(log, diff_max_bins=3, positive_diffs=True)
print('diff',diff,type(diff))
superflux_1 = np.mean(diff, axis=1)
print('superflux_1',superflux_1,type(superflux_1),superflux_1.size)

# 或者封装

class BetterSuperFluxProcessing(object):

    def __init__(self, num_bands=24, diff_max_bins=3, positive_diffs=True):
        self.num_bands = num_bands
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs

    def process(data):
        spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(data, num_bands=self.num_bands)
        diff = madmom.audio.spectrogram.SpectrogramDifference(diff, diff_max_bins=self.diff_max_bins,
                                                              positive_diffs=self.positive_diffs)
        return np.mean(diff, axis=1)
