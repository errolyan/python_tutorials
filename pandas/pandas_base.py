# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  
@Email:2681506@gmail.com   
@Date:  2019-05-22  16:32
@File：pandas_base.py
@Describe:learning pandas
'''

import pandas as pd
import numpy as np
print(pd.__version__)
print(np.__version__)

s = pd.Series([1,3,5,np.nan,6,8])
print("s:",s,type(s),id(s),"s[1]=",s[1])
print("\n")
dates = pd.date_range('20130101', periods=6)
print("dates:",dates,type(dates),id(dates),"dates[3]= ",dates[3])
print("\n")
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print("df=\n",df,df.dtypes,"\n")


df2 = pd.DataFrame({ 'A' : 1.,
                      'B' : pd.Timestamp('20130102'),
                      'C' : pd.Series(1,index=list(range(5)),dtype='float32'),
                      'D' : np.array([3] * 5,dtype='int32'),
                      'E' : pd.Categorical(["test","train","test","train","hello"]),
                      'F' : 'foo' })
print("\n",df2,df2.dtypes)
# 统计特征
data_des = df.describe()
print("df.describe()",data_des,type(data_des))
print(df.columns[1])

# 转至
df3 = df2.T
print(df3)
df4 = df.sort_index(axis=1, ascending=False)
print(df4,df)
print(df.sort_values(by='B'))
print(df.loc['20130102':'20130104',['A','C']])
df5 = df.iloc[1:3,:]
print(df5)
df6 = df[df.A > 0]

print(df2.columns)

# to get the boolean mask where values are nan
# print(pd.isna(df))
# print(df.mean())
# print(df.mean(1))
# print(df.apply(lambda x: x.max() - x.min()))