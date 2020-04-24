# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-01  13:50
'''

import madmom
import numpy as np
import matplotlib.pyplot as plt

'''
从音频文件中获得超通量起始检测函数
'''
filename = "../datasets/test.wav"
sig = madmom.audio.signal.Signal(filename)
fs = madmom.audio.signal.FramedSignal(sig)
stft = madmom.audio.stft.ShortTimeFourierTransform(fs)
spec = madmom.audio.spectrogram.Spectrogram(stft)
filt = madmom.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24)
log = madmom.audio.spectrogram.LogarithmicSpectrogram(filt)
diff = madmom.audio.spectrogram.SpectrogramDifference(log, diff_max_bins=3, positive_diffs=True)
superflux_1 = np.mean(diff, axis=1)


class BetterSuperFluxProcessing(object):

    def __init__(self, num_bands=24, diff_max_bins=3, positive_diffs=True):
        self.num_bands = num_bands
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs

    def process(data):
        spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(data, num_bands=self.num_bands)
        diff = madmom.audio.spectrogram.SpectrogramDifference(diff, diff_max_bins=self.diff_max_bins,
                                                              positive_diffs=self.positive_diffs)
        return np.mean(diff, axis=1)superflux_2 = np.mean(diff, axis=1)