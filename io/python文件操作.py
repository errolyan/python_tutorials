# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： python文件操作
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/31  13:53
-------------------------------------------------
   Change Activity:
          2020/3/31 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import os

# 获取文件名字和后缀
file_ext = os.path.splitext('more_io.py')
front,ext = file_ext
print('front =',front,'\n ext =',ext)
file_name = front.split('/')[-1]
print('file_name =',file_name)

# 读写文件夹
def mkdir(path):
    isexists = os.path.exists(path)
    if not isexists:
        os.mkdir(path)