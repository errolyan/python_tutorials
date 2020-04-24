# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： print++_example
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/1/5  16:23
-------------------------------------------------
   Change Activity:
                  2020/1/5  16:23:
-------------------------------------------------
'''
__author__ = 'yanerrol'

# 1
from termcolor import colored
print(colored('闫二乐','red'),colored('是编程人员，其次是一个算法工程师','yellow'))

import sys

text = colored('帅b','red',attrs=['reverse','blink'])
print(text)

# 2
print('\033[1;30;43m 帅b，哈哈 ')

# 3
from colorama import Fore,Back,Style
print(Fore.RED + 'some red text')
print(Back.YELLOW + 'and with a green background')
print(Style.DIM + 'and in dim text')
print(Style.RESET_ALL)
print('helloworld')