# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： osdirpath
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/27  13:47
-------------------------------------------------
   Change Activity:
          2020/3/27 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import os
dir = "test/test2/task/"
if os.path.exists(dir):
    print(dir)
else:
    os.makedirs(dir)


