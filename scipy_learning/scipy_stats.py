# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  统计学习分析
@Evn     :  
@Date    :   - 
'''
from scipy.stats import norm
import numpy as np
cdfarr = norm.cdf(np.array([1,-1,0,1,3,-2]))
print('cdfarr',cdfarr,type(cdfarr))


# 二项分布
from scipy.stats import uniform
cvar = uniform.cdf([0,1,2,3,4,5],loc =1,scale =4)
print('cvar',cvar,type(cvar))


# 统计特征分析
from scipy import stats
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9])
print ('x.describe',x.max(),x.min(),x.mean(),x.var())
