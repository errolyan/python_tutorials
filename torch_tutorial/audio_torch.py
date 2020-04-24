# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''

import torch
import torchaudio
import matplotlib.pyplot as plt

file_path = './data/000-test.wav'
waveform,sample_rate = torchaudio.load(file_path)
print('waveform',waveform,type(waveform),'\nsample_rate:',sample_rate,type(sample_rate))
print("shape of waveform:{}".format(waveform.size))
print("sample rate of waveform:{}".format(sample_rate))
print("waveform.t(){}".format(waveform.t()))
print('type(waveform.t()))',type(waveform.t()))

print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(waveform.min(), waveform.max(), waveform.mean()))
# 正则化音频数据
def normalize(waveform):
    tensor_minusmean = waveform - waveform.mean()
    return tensor_minusmean/(tensor_minusmean.abs().max())

plt.figure()
plt.plot(waveform.t().numpy())
#plt.show()

# 声谱图
specgram = torchaudio.transforms.Spectrogram()(waveform)
print('specgram',specgram,type(specgram),specgram.size())

log_specgram = specgram.log2()
print('log_specgram',log_specgram,type(log_specgram),log_specgram.size())

log_specgram = log_specgram[0,:,:]
print('log_specgram2',log_specgram,type(log_specgram),log_specgram.size())

numpy_specgram = log_specgram.numpy()
print('numpy_specgram',numpy_specgram,type(numpy_specgram))

# plt.figure()
# # plt.imshow(numpy_specgram,cmap ="gray")
# # plt.show()

# mel 梅尔谱图
melspecgram = torchaudio.transforms.MelSpectrogram()(waveform)
print('melspecgram',melspecgram,type(melspecgram),melspecgram.size())

log_specgram = melspecgram.log2()
print('log_mel_specgram',log_specgram,type(log_specgram),log_specgram.size())

log_specgram = log_specgram[0,:,:].detach()
print('log_specgram2',log_specgram,type(log_specgram),log_specgram.size())

numpy_specgram = log_specgram.numpy()
print('numpy_specgram',numpy_specgram,type(numpy_specgram))

# plt.figure()
# plt.imshow(numpy_specgram,cmap ="gray")
# plt.show()

# 重采样
new_sampel_rate = sample_rate / 2
channels = 0
transformed = torchaudio.transforms.Resample(sample_rate,new_sampel_rate)(waveform[channels,:].view(1,-1))
print('transformed',transformed, type(transformed),transformed.size())
# plt.figure()
# plt.plot(transformed[0,:].numpy())
# plt.show()

# 编码音频数据
transformed11 = torchaudio.transforms.MuLawEncoding()(waveform)
print('transformed33',transformed11, type(transformed11),transformed11.size())

transformed = transformed11[0,:]
print('transformed44',transformed, type(transformed),transformed.size())
# plt.figure()
# plt.plot(transformed.numpy())
# plt.show()

# 解码
restructed = torchaudio.transforms.MuLawDecoding()(transformed11)
print('restructed',restructed, type(restructed),restructed.size())
print('waveform',waveform, type(waveform),waveform.size())
plt.figure()
plt.plot(restructed[0,:].numpy())
# plt.show()

# 对比编码前和解码后的音频差异
err = ((waveform -restructed).abs() / waveform.abs()).median()
print("差异:{:.2%}".format(err),type(err),err.size())

# 加载kaldi

n_fft = 400.0
frame_length = n_fft / sample_rate * 1000.0
frame_shift = frame_length / 2.0

params = {
    "channel": 0,
    "dither": 0.0,
    "window_type": "hanning",
    "frame_length": frame_length,
    "frame_shift": frame_shift,
    "remove_dc_offset": False,
    "round_to_power_of_two": False,
    "sample_frequency": sample_rate,
}

specgram = torchaudio.compliance.kaldi.spectrogram(waveform, **params)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.t().numpy(), cmap='gray')
plt.show()

# We also support computing the filterbank features from waveforms, matching Kaldi’s implementation.
fbank = torchaudio.compliance.kaldi.fbank(waveform, **params)

print("Shape of fbank: {}".format(fbank.size()))

plt.figure()
plt.imshow(fbank.t().numpy(), cmap='gray')
