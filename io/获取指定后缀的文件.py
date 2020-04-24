# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： 获取指定后缀的文件
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/31  14:14
-------------------------------------------------
   Change Activity:
          2020/3/31 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import os

import os
def find_file(work_dir,extension='jpg'):
    lst = []
    for filename in os.listdir(work_dir):
        print(filename)
        splits = os.path.splitext(filename)
        ext = splits[1] # 拿到扩展名
        if ext == '.'+extension:
            lst.append(filename)
    return lst
r = find_file('.','py')
print(r) # 返回所有目录下的 md 文件


