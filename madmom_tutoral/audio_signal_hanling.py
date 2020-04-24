# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： audio_signal_hanling
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/4/19  22:43
-------------------------------------------------
   Change Activity:
          2020/4/19 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import numpy as np
import matplotlib.pyplot as plt
import madmom

signal,sample_rate = madmom.audio.signal.load_wave_file('data/sample.wav')
print('sample',signal,type(signal),'\n sample rate',sample_rate)

# 读取音频数字信号
sig = madmom.audio.signal.Signal('data/sample.wav')
print('sig',sig,'\n sample_rate',sig.sample_rate)

# 将信号重采样
framedsignal = madmom.audio.signal.FramedSignal(sig,frame_size=2048,hop_size=441)
print('fs',framedsignal,framedsignal.frame_size,'\n [0]',framedsignal[0],'\n [10]',framedsignal[10] ,
      '\n fps',framedsignal.fps,'\n num_frames',framedsignal.num_frames,'hop_size',framedsignal.hop_size)


fs = madmom.audio.signal.FramedSignal(sig, frame_size=2048, fps=200)
print('fs',fs,fs.frame_size,'\n [0]',fs[0],'\n [10]',fs[10] ,
      '\n fps',fs.fps,'\n num_frames',fs.num_frames,'fs.hop_size',fs.hop_size)


# STFT
stft = madmom.audio.stft.STFT(fs)
print('stft',stft,type(stft),stft[0])

# 频谱图
spec = madmom.audio.spectrogram.Spectrogram(stft)
print('spec',spec,type(spec),'\nshape',spec.shape)
plt.imshow(spec[:, :200].T, aspect='auto', origin='lower')
plt.show()

print(spec.stft.frames.overlap_factor)

from scipy.ndimage.filters import maximum_filter

spec = madmom.audio.spectrogram.Spectrogram('data/sample.wav')
print('spec',spec,type(spec))


# calculate the difference
diff = np.diff(spec, axis=0)
# keep only the positive differences
pos_diff = np.maximum(0, diff)
# sum everything to get the spectral flux
sf = np.sum(pos_diff, axis=1)

plt.figure()
plt.imshow(spec[:, :200].T, origin='lower', aspect='auto')
plt.show()

plt.figure()
plt.imshow(pos_diff[:, :200].T, origin='lower', aspect='auto')
plt.show()

plt.figure()
plt.plot(sf)
plt.show()

## second method
sf = madmom.features.onsets.spectral_flux(spec)

filt_spec = madmom.audio.spectrogram.FilteredSpectrogram(spec, filterbank=madmom.audio.filters.LogFilterbank,
                                                         num_bands=24)
plt.imshow(filt_spec.T, origin='lower', aspect='auto')
plt.show()


log_spec = madmom.audio.spectrogram.LogarithmicSpectrogram(filt_spec, add=1)
plt.imshow(log_spec.T, origin='lower', aspect='auto')
plt.show()


# maximum filter size spreads over 3 frequency bins
size = (1, 3)
max_spec = maximum_filter(log_spec, size=size)
plt.imshow(max_spec.T, origin='lower', aspect='auto')
plt.show()


# sum everything to get the onset detection function
superflux = np.sum(pos_diff, axis=1)

plt.figure()
plt.plot(superflux)
plt.show()