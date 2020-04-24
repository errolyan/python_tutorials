# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  scipy io
@Evn     :  
@Date    :   - 
'''
import scipy.io as sio
import numpy as np

vect = np.arange(10)
sio.savemat('array.mat',{'vect':vect})
mat_file_content = sio.loadmat('array.mat')
print(mat_file_content)

'''
# 线性代数
# 求解线性代数
'''

from scipy import linalg
import numpy as np
# ax = b
a = np.array([[3,2,0],[1,-1,0],[0,5,1]])
b = np.array([2,4,-1])

x = linalg.solve(a,b)
print('x',x,x.shape)

'''查找行列式'''
from scipy import linalg
import numpy as np

# 计算方阵的值
A = np.array([[1,2],[3,4]])
x = linalg.det(A)
print('x',x)

'''
特征值特征向量
'''
from scipy import linalg
import numpy as np

A = np.array([[1,2],[3,4]])
# 计算特征值和特征向量
l,v = linalg.eig(A)
print('特征值=',l,'\n特征向量=',v)

'''
奇异值分解
'''
from scipy import linalg
import numpy as np

a = np.random.randn(3,2) + 1.j*np.random.randn(3,2)
U,s,vh = linalg.svd(a)
print('U',U,'\nvh',vh,'\ns',s)