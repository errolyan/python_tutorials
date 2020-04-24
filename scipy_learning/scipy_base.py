# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  scipy like MATLAB
numpy scipy matplotlib
@Evn     :  
@Date    :   - 
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg,optimize
from scipy.cluster.vq import kmeans,vq,whiten

import scipy
print(scipy.__version__)

# 常量
from scipy import constants as C
print (C.c) # 真空中的光速
print (C.h) # 普朗克常数


print(np.info(optimize.fmin))
from numpy import vstack,array
from numpy.random import rand

# data generation with three features
data = vstack((rand(100,3) + array([.5,.5,.5]),rand(100,3)))
print(data,type(data))
# 美化数据
data = whiten(data)
print(data)
# computing K-Means with K = 3 (2 clusters)
centroids,fff= kmeans(data,3)
print(centroids,'\n',fff)
# 给每个代码分配一个集群
clx,_ = vq(data,centroids)

print('clx',clx.shape,data.shape)