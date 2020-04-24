# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： 拼接
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/4/18  15:23
-------------------------------------------------
   Change Activity:
          2020/4/18 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import pandas  as pd
import numpy as np

##### merge
# 内连接
df1 = pd.DataFrame({'alpha':['A','B','B','C','D','E'],'feature1':[1,1,2,3,3,1], 'feature2':['low','medium','medium','high','low','high']})
df2 = pd.DataFrame({'alpha':['A','A','B','F'],'pazham':['apple','orange','pine','pear'], 'kilo':['high','low','high','medium'],'price':np.array([5,6,5,7])})
print(df1)
print(df2)

df3 = pd.merge(df1, df2, how='inner',on = 'alpha')
print('df3',df3)


# 外连接
df4 = pd.merge(df1, df2, how='outer',on = 'alpha')
print('df4',df4)

# 右连接
df5 = pd.merge(df1, df2, how='right',on = 'alpha')
print('df5',df5)


#### join
