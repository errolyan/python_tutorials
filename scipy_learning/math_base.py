# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''

'''
1.奇异分解
'''
import numpy as np
from scipy.linalg import svd

A = np.array([[1,2],[3,4],[6,7]])
print("A",A,type(A),A.size)

U,S,V = svd(A)
print("U",U,type(U))
print("S",S,type(S))
print("V",V,type(V))

print(U.max())
'''
2.scipy base

'''
import scipy
print(scipy.__file__)
print(scipy.__version__)

import numpy as np
from scipy import io as sio
array = np.ones((4, 4))
sio.savemat('example.mat', {'ar': array})
data = sio.loadmat('example.mat', struct_as_record=True)
print(data['ar'])
import scipy.special
print(help(scipy.special))