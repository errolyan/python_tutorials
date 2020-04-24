# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  scipy 常量
@Evn     :  
@Date    :   - 
'''
from scipy import constants
import math

print('scipy - pi = %.16F'%constants.pi)
print('math - pi = %.16f'%math.pi)

import scipy.constants
res = scipy.constants.physical_constants['alpha particle mass']
print(res)

# 快速傅里叶变换
from scipy.fftpack import fft,ifft
import numpy as np
x = np.array([1,2,1,-1,1.5])
y = fft(x)
print('y',y)

# 逆傅里叶变换
yinv = ifft(y)
print('yinv',yinv)

import numpy as  np
time_step = 0.02
period = 5
time_vec = np.arange(0,20,time_step)
sig =np.sin(2*np.pi/period*time_vec + 0.5*np.random.rand(time_vec.size))
print('sig.size',sig.size)

# 信号傅里叶变换
from scipy import fftpack
samples = fftpack.fftfreq(sig.size,d=time_step)
sig_fft = fftpack.fft(sig)
print('sig_fft',sig_fft)

# 离散余弦函数
from scipy.fftpack import dct,idct
mydict = dct(np.array([4,3,5,10,5,3]))
print('mydict',mydict)
data =  idct(mydict)
print('data',data)