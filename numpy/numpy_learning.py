# # -*- coding:utf-8 -*-
# # /usr/bin/python
# '''
# @Author  :  Yan Errol
# @Describe:  numpy
# @Evn     :
# @Date    :  2019-07-10  17:11
# '''
#
# import numpy as np
#
# data = np.array([1,2,3])
# print(data,type(data))
# print(data.max())
# data1 = np.ones(3)
# print(data1,type(data1))
# print(np.random.random(3))
#
# print("A"*3)
#
# from mxnet import ndarray
#
# x = ndarray.arange(12)
# print(x,type(x),x.shape,x.size)
# X = x.reshape((3,4))
# print(X,type(X),X.shape,X.size)
#
# x3 = ndarray.zeros((1,2,3,4))
# print(x3,type(x3),x3.shape,x3.size)
#
# Y = ndarray.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(Y)
# print(X + Y)
# print(X * Y)
# print(Y.exp())
# print(ndarray.dot(X ,Y.T)) # Y.T 是转置 ndarange.dot是矩阵相乘
# print(ndarray.concat(X,Y,dim = 0))
# print(ndarray.concat(X,Y,dim = 1))
# print(X == Y) # XY位置相同的相等就是1，否则是0
# Z = X.sum()
# print(Z,Z.shape,Z.size)
#
# A = ndarray.arange(3).reshape((3,1))
# B = ndarray.arange(2).reshape((1,2))
# print(A , B)
# C = A + B # 广播机制
# print(C)
#
# print(X[1:3,1:2])
# print(id(X))
# X[2:3,:] = 0
# print(X)
# print(id(X))
# print(id(B))
# B = A+B
# print(id(B))
#
# print(ndarray.array(data)) #np 2 ndarray
# print()
# P = np.ones((2, 3))
# D = ndarray.array(P)
# print(D.asnumpy())
#

'''
自动梯度
'''
from mxnet import autograd, nd
x = nd.arange(4).reshape((4, 1))
print(x)
x.attach_grad()
with autograd.record():
    y = 2 * nd.dot(x.T, x)

y.backward()

assert (x.grad - 4 * x).norm().asscalar() == 0
print(x.grad)

print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())

