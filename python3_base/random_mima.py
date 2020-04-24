# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： random_mima
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/8  03:11
-------------------------------------------------
   Change Activity:
                  2020/2/8  03:11:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import random

alphabet = "abcdefghijklmnopqrstuvwxyz"
pw_length = 8
mypw = ""

for _ in range(pw_length):
    next_index = random.randrange(len(alphabet))
    mypw = mypw + alphabet[next_index]

# 替换一个或两个字母为数字
for _ in range(random.randrange(1, 3)):
    replace_index = random.randrange(len(mypw) // 2)
    mypw = mypw[0:replace_index] + str(random.randrange(10)) + mypw[replace_index + 1:]

# 替换一个或两个字母为大写
for _ in range(random.randrange(1, 3)):
    replace_index = random.randrange(len(mypw) // 2, len(mypw))
    mypw = mypw[0:replace_index] + mypw[replace_index].upper() + mypw[replace_index + 1:]
print(mypw)