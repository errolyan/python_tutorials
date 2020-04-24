# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  音频处理
@Evn     :  pip install librosa
@Date    :  2019-08-04  15:23
'''

'''
pip install librosa
ffmpeg is very stronger.

librosa.beat:用于检测速度和节拍
librosa.core:用于从磁盘加载音频和计算各种频谱图
librosa.decompose:实现矩阵分解进行谐波和冲击源分离通用频谱图分解
librosa.display:音频特征的显示
librosa.effects:时域音频处理，音高变换和时间拉伸，时域包装器。
librosa.feature:特征提取和操作：色度图，伪常数Q（对数频率）变换，Mel频谱图，MFCC和调谐估计
librosa.filters:滤波器生成色度。伪CQT、CQT等
librosa.onset:其实检测和起始强度计算。
librosa.segment:用于结构分段的函数
librosa.swquence:顺序建模功能
librosa.util:辅助工具（规范化。填充、居中）
'''
# Beat tracking example
#from __future__ import print_function
import librosa
from librosa import display
import matplotlib.pyplot as plt
# 1. Get the file path to the included audio example
filename = librosa.util.example_audio_file()

# 2. Load the audio as a waveform `y`data，sampling_rate
#    Store the sampling rate as `sr`
filename = "a.wav"
y, sr = librosa.load(filename)
plt.figure(figsize=(12, 4))
display.waveplot(y,sr=sr)
plt.show()
plt.savefig('./test.png')

print("y",y,"\nsr",sr)

# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print("tempo",tempo, "\nbeat_frames",beat_frames)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# 5.Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)
print(y_harmonic,'\n', y_percussive)


