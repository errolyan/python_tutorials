# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import numpy as np
print('np version:',np.__version__)
print(np.show_config())




num1 = np.array([[1,2,3],[6,7,8]])
# 如何计算array在内存中的大小
print('num1的大小为：%d bytes'%(num1.size*num1.itemsize))


num2 = num1
num1[0,2] = 0
num2[1][1] = 9
print('num1',num1,type(num1),num1.size,num1.shape,id(num1),id(num2),num1 == num2,num2[0],num2[1])

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

# 加法
sum_a_b = a + b
print('sum_a_b',sum_a_b,type(type(sum_a_b)),sum_a_b.size,sum_a_b.shape)

# 乘法
product = a * b
print('product',product,type(type(product)),product.size,product.shape)

# 除法
quotient = a / b
print('quotient',quotient,type(type(quotient)),quotient.size,quotient.shape)

print('quotient.max()',quotient.max())
print('quotient.min()',quotient.min())
print('quotient.sum()',quotient.sum())

# 点乘
matrix_product = a.dot(b)
print('matrix_product',matrix_product,type(matrix_product),matrix_product.size,id(matrix_product))

a_a = a[0,0:1]
print('a_a',a_a,type(a_a),a_a.size,a_a.dtype,a_a.shape,a_a.ndim)

# 特殊的判断符号
print('np.nan == np.nan',np.nan == np.nan)
print('0*np.nan',0*np.nan)
print(np.inf > np.nan)
print('np.nan - np.nan:',np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)

# 标准化数据
Z = np.random.random((5,5))
Z = (Z - np.mean (Z)) / (np.std (Z))
print('标准化Z',Z)

color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])

print(color,type(color),color.shape, color.ndim)

# 点乘
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

# Alternative solution, in Python 3.5 and above
Z = np.ones((5,3)) @ np.ones((3,2))
print(Z)

Z = np.arange(11)
print('Z**Z',Z**Z)
print('2 << Z >> 2',2 << Z >> 2)

Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)

print(sum(range(5),-1))
print('range(5),-1',range(5),-1)

print(np.sum(range(5),-1))

def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)

Z = np.random.random(10)
Z.sort()
print('z',Z)

Z = np.arange(10)
print('np.add.reduce(Z)',np.add.reduce(Z))

Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print(Z)

X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5])
print(confint)

X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])

# 字符操作
x1 = np.array(['Hello ',' Say'],dtype = np.str)
x2 = np.array(['world',' something'],dtype = np.str)
out = np.char.add(x1,x2)
print(out)

x = np.array(['Hello ','Say '],dtype = np.str)
out = np.char.multiply(x,3)
print(out)

x = np.array(['34'], dtype=np.str)
out = np.char.zfill(x, 4)
print(out)