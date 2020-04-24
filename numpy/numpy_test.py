# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-10-12  16:07
'''
import numpy as np
array = np.array([
    [1,3,5],
    [4,6,9]
])

print(array)

# 维度
print('number of dim:', array.ndim)
# 行列数
print('shape:',array.shape)
# 元素个数
print('size:',array.size)

# 一维array
a = np.array([2,23,4], dtype=np.int32) # np.int默认为int32
print(a)
print(a.dtype)


# 一维矩阵运算
a = np.array([10,20,30,40])
b = np.arange(4)
print(a,b)

c = a - b
print(c)

d = a*b
print('d',d)
print(a.dot(b)) #对应相乘再相加

print(b**2)


a = np.random.random((2,4))
print(np.sum(a))
print(np.max(a),np.min(a))

print("sum=",np.sum(a,axis=1)) # 按行
print('sum = ',np.sum(a,axis=0)) # 按列

# 基本运算
A = np.arange(2,14).reshape(3,4)
print(A)

# 最小元素索引
print(np.argmin(A))

# 最大元素索引
print(np.argmax(A))

# 均值
print(np.mean(A),np.average(A))

# 中位数
print(np.median(A))

# 累加
print(np.cumsum(A))

A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B)))

C = np.vstack((A,B))
print(C)