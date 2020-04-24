# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  让python的效率提升
@Evn     :  
@Date    :  2019-08-31  17:32
'''

import numba as nb
import numpy as np
import time

from numba import jit

@jit(nopython=True) # jit ,numba装饰器的一种
def go_fast(a):

    trace =0
    for i in range(a.shape[0]): # numba 擅长处理循环
        trace += np.tanh(a[i,i])


    return a + trace
x = np.arange(1000000).reshape(1000, 1000)
start_time = time.time()
go_fast(x)
end_time = time.time()
print('end_time-start_time',end_time-start_time)



def go_fast1(a):
    trace = 0
    for i in range(a.shape[0]):  # numba 擅长处理循环
        trace += np.tanh(a[i, i])

    return a + trace


x = np.arange(1000000).reshape(1000, 1000)
start_time = time.time()
go_fast1(x)
end_time = time.time()
print('end_time-start_time1', end_time - start_time)