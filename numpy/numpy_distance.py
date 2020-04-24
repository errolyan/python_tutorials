# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： numpy_distance
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/4/16  15:11
-------------------------------------------------
   Change Activity:
          2020/4/16 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import numpy as np

# 欧式距离 Euclidean distance
def euclidean(x, y):
    return np.sqrt(np.sum((x-y)**2))

x = np.array([1,2,3])

y = np.array([2,3,4])
print('x',x,'y',y)
distance = euclidean(x,y)
print('euclidean:',distance)

# 马哈顿距离 manhattan distance ：绝对值距离

def Manhattan(x,y):
    return np.sum(np.abs(x-y))

distance = Manhattan(x,y)
print('Manhattan:',distance)


# 切比雪夫距离 Chebyshev distance
def chebyshev(x, y):
    return np.max(np.abs(x - y))

distance = chebyshev(x,y)
print('chebyshev:',distance)

# 闵可夫斯基距离(Minkowski distance)

def minkowski(x, y, p):

    return np.sum(np.abs(x - y) ** p) ** (1 / p)

distance = minkowski(x,y,1)
print('minkowski:',distance)


# 汉明距离
def hamming(x, y):
    return np.sum(x != y) / len(x)
distance = hamming(x,y)
print('hamming:',distance)