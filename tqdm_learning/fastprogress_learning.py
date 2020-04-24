# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： fastprogress_learning
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/4/15  16:09
-------------------------------------------------
   Change Activity:
          2020/4/15 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

from fastprogress.fastprogress import master_bar, progress_bar
from time import sleep
mb = master_bar(range(10))
for i in mb:
    for j in progress_bar(range(100), parent=mb):
        sleep(0.01)
        mb.child.comment = 'second bar stat'
    mb.first_bar.comment = 'first bar stat'
    mb.write('Finished loop {i}.')


