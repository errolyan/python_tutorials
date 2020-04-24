# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： example2
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/31  15:08
-------------------------------------------------
   Change Activity:
                  2019/12/31  15:08:
-------------------------------------------------
'''
__author__ = 'yanerrol'
from multiprocessing.pool import Pool

def hhh(i):
    return i * 2


if __name__ == '__main__':
    pool = Pool(processes=2)
    lst = range(100)
    hh = pool.map(hhh, lst)
    print(hh)