# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： example_numpy
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/6  11:33
-------------------------------------------------
   Change Activity:
                  2020/2/6  11:33:
-------------------------------------------------
'''
__author__ = 'yanerrol'

from memory_profiler import profile
import numpy as np
import scipy.signal


@profile
def create_data():
    ret = []
    for n in range(70):
        ret.append(np.random.randn(1, 70, 71, 72))
    return ret


@profile
def process_data(data):
    data = np.concatenate(data)
    detrended = scipy.signal.detrend(data, axis=0)
    return detrended


if __name__ == "__main__":
    data1 = create_data()
    data2 = process_data(data1)