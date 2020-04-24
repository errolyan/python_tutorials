# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  学习madmom
@Evn     :  pip install madmom
@Date    :  2019-09-01  12:57
'''

import numpy as np
import matplotlib.pyplot as plt

import madmom
filename = "../datasets/test.wav"
signal, sample_rate = madmom.audio.signal.load_wave_file(filename)
print('signal',signal,'sample_rate' ,sample_rate)
sig = madmom.audio.signal.Signal(filename)
sample_rate = sig.sample_rate
print('sig',sig,'sample_rate',sample_rate)

fs = madmom.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
print('fs',fs)
print(fs.frame_rate, fs.hop_size, fs[0])

stft = madmom.audio.stft.STFT(fs)
print(stft[0:2])
spec = madmom.audio.spectrogram.Spectrogram(stft)
plt.figure()
plt.imshow(spec[:, :200].T, aspect='auto', origin='lower')
plt.savefig('./test.png')

# calculate the difference
diff = np.diff(spec, axis=0)
# keep only the positive differences
pos_diff = np.maximum(0, diff)
# sum everything to get the spectral flux
sf = np.sum(pos_diff, axis=1)

plt.figure()
plt.imshow(spec[:, :200].T, origin='lower', aspect='auto')
plt.savefig('./test1.png')

plt.figure()
plt.imshow(pos_diff[:, :200].T, origin='lower', aspect='auto')
plt.savefig('./test2.png')

plt.figure()
plt.plot(sf)
plt.savefig('./test3.png')

filt_spec = madmom.audio.spectrogram.FilteredSpectrogram(spec, filterbank=madmom.audio.filters.LogFilterbank,
                                                         num_bands=24)
plt.imshow(filt_spec.T, origin='lower', aspect='auto')
plt.figure()
log_spec = madmom.audio.spectrogram.LogarithmicSpectrogram(filt_spec, add=1)
plt.imshow(log_spec.T, origin='lower', aspect='auto')
plt.savefig('./test4.png')
