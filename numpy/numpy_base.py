# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  @Email:2681506@gmail.com   
@Date:  2019-06-09  10:38
@Describe:numpy
@Evn:
'''

import numpy as np

A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,3,4])[:,np.newaxis]
C = np.vstack((A,B))
D = np.hstack((A,B))

print(C,D)
print('A',A.shape,A.ndim,A.size,A.dtype)
A = np.reshape(A,(3,))
print('A',A.shape,A.ndim,A.size,A.dtype)
E = np.concatenate((A,B),axis=1)
print("E",E)

F = np.arange(12).reshape((3,4))
print("F",F)
print(np.split(F,3,axis=0))

A = np.array([1,1,1])
D = A
F = A.copy()
F[2] = 100
print("F",F)
print("A",A)
print("D",D)

# 基本操作
array = np.array([[1,3,5],[4,6,9]])
print(array) # 列表转化为矩阵

print('number of dim:', array.ndim) # 维度
print('shape:',array.shape) # 行列数
print('size:',array.size) # 元素个数

a = np.array([2,23,4], dtype=np.int32) # np.int默认为int32
print(a)
print(a.dtype)

# 多维array
a = np.array([[2,3,4],
              [3,4,5]])
print(a) # 生成2行3列的矩阵

a = np.zeros((3,4))
print(a) # 生成3行4列的全零矩阵

# 创建全一数据，同时指定数据类型
a = np.ones((3,4),dtype=np.int)
print(a)

# 创建全空数组，其实每个值都是接近于零的数
a = np.empty((3,4))
print(a)

# 创建连续数组
a = np.arange(10,21,2) # 10-20的数据，步长为2
print(a)

# 使用reshape改变上述数据的形状
b = a.reshape((2,3))
print(b)

# 创建线段型数据
a = np.linspace(1,10,20) # 开始端1，结束端10，且分割成20个数据，生成线段
print(a)

# 同时也可以reshape
b = a.reshape((5,4))
print(b)

# 一维矩阵运算
a = np.array([10,20,30,40])
b = np.arange(4)
print(a,b)

c = a - b
print(c)

print(a*b)

c = b**2
print(c)

c = np.sin(a)
print(c)

print(b<2)

a = np.array([1,1,4,3])
b = np.arange(4)
print(a==b)

a = np.array([[1,1],[0,1]])
b = np.arange(4).reshape((2,2))
print(a)

# 多维度矩阵乘法
# 第一种乘法方式:
c = a.dot(b)
print(c)

# 第二种乘法:
c = np.dot(a,b)
print(c)

A = np.arange(2,14).reshape((3,4))
print(A)

# 最小元素索引
print(np.argmin(A)) # 0

# 最大元素索引
print(np.argmax(A)) # 11

# 求整个矩阵的均值
print(np.mean(A)) # 7.5

print(np.average(A)) # 7.5

print(A.mean()) # 7.5

# 中位数
print(np.median(A)) # 7.5

# 累加
print(np.cumsum(A))

# 累差运算
B = np.array([[3,5,9],
              [4,8,10]])
print(np.diff(B))

C = np.array([[0,5,9],
              [4,0,10]])
print(np.nonzero(B))
print(np.nonzero(C))

A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B)))

C = np.vstack((A,B))
print(C)

print(A.shape,B.shape,C.shape)# 从shape中看出A,B均为拥有3项的数组(数列)

# horizontal stack左右合并
D = np.hstack((A,B))
print(D)

print(A[np.newaxis,:]) # [1 1 1]变为[[1 1 1]]

print(A[np.newaxis,:].shape) # (3,)变为(1, 3)

# concatenate的第一个例子
print("------------")
print(A[:,np.newaxis].shape) # (3,1)

A = A[:,np.newaxis] # 数组转为矩阵
B = B[:,np.newaxis] # 数组转为矩阵

# axis=0纵向合并
C = np.concatenate((A,B,B,A),axis=0)
print(C)

# axis=1横向合并
C = np.concatenate((A,B),axis=1)
print(C)

# concatenate的第二个例子
print("-------------")
a = np.arange(8).reshape(2,4)
b = np.arange(8).reshape(2,4)
print(a)
print(b)
print("-------------")

# axis=0多个矩阵纵向合并
c = np.concatenate((a,b),axis=0)
print(c)

# axis=1多个矩阵横向合并
c = np.concatenate((a,b),axis=1)
print(c)

A = np.arange(12).reshape((3,4))
print(A)

# 等量分割
# 纵向分割同横向合并的axis
print(np.split(A, 2, axis=1))

# 横向分割同纵向合并的axis
print(np.split(A,3,axis=0))

'''
画乌龟
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
def compute_mandelbrot(N_max, some_threshold, nx, ny): # A grid of c-values
    x = np.linspace(-2, 1, nx)
    y = np.linspace(-1.5, 1.5, ny)
    c = x[:,newaxis] + 1j*y[newaxis,:] # Mandelbrot iteration
    z=c
    for j in range(N_max):
        z = z**2 + c
    mandelbrot_set = (abs(z) < some_threshold)
    return mandelbrot_set
mandelbrot_set = compute_mandelbrot(50, 50., 601, 401)
plt.imshow(mandelbrot_set.T, extent=[-2, 1, -1.5, 1.5])
plt.gray()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)
plt.axes([0.025, 0.025, 0.95, 0.95])
plt.scatter(X,Y, s=75, c=T, alpha=.5)
plt.xlim(-1.5, 1.5)
plt.xticks(())
plt.ylim(-1.5, 1.5)
plt.yticks(())
plt.show()
