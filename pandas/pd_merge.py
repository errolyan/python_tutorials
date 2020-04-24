# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import pandas as pd

df1 = pd.read_csv('./data/1.csv')
print('df1',df1)
df2 = pd.read_csv("./data/input_data_3.csv")
print("df2", df2)
df3 = pd.merge(df1,df2, on =['f','f'])
print("df3", df3)