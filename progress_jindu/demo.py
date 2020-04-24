# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： demo
   Description :  进度条
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/3  22:33
-------------------------------------------------
   Change Activity:
                  2020/2/3  22:33:
-------------------------------------------------
'''
__author__ = 'yanerrol'
from progress.bar import Bar

bar = Bar('progresssing',max=20)
for i in range(20):
    bar.next()
bar.finish()