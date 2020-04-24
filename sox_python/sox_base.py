# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： sox_base
   Description :  AIM: sox 音频处理
                  Functions: 1. 音频裁剪
                             2. 音频拼接
   Envs        :  python == 3.5
                  1. MAC:brew install sox (support for mp3, flac, or ogg files :brew install sox --with-lame --with-flac --with-libvorbis)
                  1.2 Linux : apt-get install sox
                  2. pip install sox -i https://pypi.douban.com/simple
   Author      :  yanerrol
   Date        ： 2020/2/24  09:05
-------------------------------------------------
   Change Activity:
          2020/2/24 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import sox
print(sox.__version__)
# 拆分
# create trasnformer
tfm = sox.Transformer()
# trim the audio between 5 and 10.5 seconds.
tfm.trim(5, 10.5)
# apply compression
tfm.compand()
# apply a fade in and fade out
tfm.fade(fade_in_len=1.0, fade_out_len=0.5)
# create the output file.
tfm.build('1.wav', 'audio.wav')
# see the applied effects
tfm.effects_log

# 合并
# create combiner
cbn = sox.Combiner()
# pitch shift combined audio up 3 semitones
cbn.pitch(3.0)
# convert output to 8000 Hz stereo
cbn.convert(samplerate=8000)
# create the output file
cbn.build(
    ['1.wav', 'audio.wav'], 'output.wav', 'concatenate'
)


