# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： tqdm_example
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/6  17:51
-------------------------------------------------
   Change Activity:
                  2020/2/6  17:51:
-------------------------------------------------
'''
__author__ = 'yanerrol'

from tqdm import tqdm
import time

tqdm.pandas()


text = ""
for char in tqdm(["a", "b", "c", "d"]):
    time.sleep(0.25)
    text = text + char

from tqdm import tqdm
import time
for i in tqdm(range(10000)):
    time.sleep(0.01)
    pass


