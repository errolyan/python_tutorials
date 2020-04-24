# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： example
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/6  11:31
-------------------------------------------------
   Change Activity:
                  2020/2/6  11:31:
-------------------------------------------------
'''
__author__ = 'yanerrol'

from memory_profiler import profile
# 采用装饰器的方式引用，不影响现有代码
@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()