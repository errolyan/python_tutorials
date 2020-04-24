# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  scipy optimize 优化函数
@Date    :   - 
'''
import numpy as np
from scipy.optimize import minimize,root

# 最小二乘法
def fun_rosenbrock(x):
   return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

from scipy.optimize import least_squares
input = np.array([2, 2])
res = least_squares(fun_rosenbrock, input)

print (res)


# 求根
def func(x):
    return x*2 + 2*np.cos(x)
sol = root(func,0.3)
print('sol',sol)