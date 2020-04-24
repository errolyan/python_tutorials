# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： example
   Description :  AIM: example
                  Functions: 1. python 下载
                             2. 
   Envs        :  python == 3.5
                  pip install wget -i https://pypi.douban.com/simple
   Author      :  yanerrol
   Date        ： 2020/4/14  22:54
-------------------------------------------------
   Change Activity:
          2020/4/14 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import wget
url = 'http://www.futurecrew.com/skaven/song_files/mp3/razorback.mp3'
filename = wget.download(url)

print(filename)
