# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol  Email:2681506@gmail.com   Wechat:qq260187357
Date:2019-05-13--08:45
File：numpy_learning.py
Describe:学习numpy
'''

import numpy as np

a = np.array([[1,2.1],[2,3],[4,5],[7,7]])
print(a,a.ndim,a.shape,a.size,type(a),a.dtype)
a[3][0] = 0
a[3,1] = 9
print(a,a.ndim,a.shape,a.size,type(a),a.dtype)
b = np.zeros((4,2))
print(b,b.ndim,b.shape,b.size,b.dtype)
c = np.random.random((4,2))
print(c,c.ndim,c.shape,c.size,type(c),c.dtype)
d = a[:1,1:2]
print(d,d.ndim,d.shape,d.size,d.dtype)

# 切片访问数组
row_r1 = a[1,:]
print(row_r1,row_r1.ndim,row_r1.shape,row_r1.size)
row_r2 = a[1:2, :]
print(row_r2,row_r2.ndim,row_r2.shape,row_r2.size)
e = np.add(a ,c)
print(e,e.ndim,e.shape,e.size,e.dtype)
e = np.subtract(a ,c)
print(e,e.ndim,e.shape,e.size,e.dtype)
e = np.multiply(a ,c)
print(e,e.ndim,e.shape,e.size,e.dtype)
print(e.T)

'''
广播是一种强有力的机制，可以让不同大小的矩阵进行数学计算。我们常常会有一个小的矩阵和一个大的矩阵，然后我们会需要用小的矩阵对大的矩阵做一些计算。
'''
x = np.array([[1,2,3], [4,5,6], [7,8,9]])
v = np.array([1, 0, 1])
y = np.empty_like(x)
for i in range(3):
   y[i, :] = x[i, :] + v
print(y)

v = np.array([1,2,3])  # v 的shape (3,)
w = np.array([4,5])
print(np.reshape(v, (3, 1)) * w)

x = np.array([[1,2,3], [4,5,6]])
print(x,x.ndim,x.shape,x.size,x.dtype,x + v)