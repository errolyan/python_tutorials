# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： example
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/6  11:50
-------------------------------------------------
   Change Activity:
                  2020/2/6  11:50:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import ffmpeg

(
    ffmpeg
    .input('FaceTime', format='avfoundation', pix_fmt='uyvy422', framerate=30)
    .output('out.mp4', pix_fmt='yuv420p', vframes=1000)
    .run()
)