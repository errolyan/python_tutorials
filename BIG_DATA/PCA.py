# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  @Email:2681506@gmail.com   
@Date:  2019-06-06  10:21
@Describe:one_hot
@Evn:
'''

import pandas as pd
from sklearn import linear_model
df = pd.DataFrame({'City': ['SF', 'SF', 'SF', 'NYC', 'NYC', 'NYC','Seattle', 'Seattle', 'Seattle'],
				   'Rent': [3999, 4000, 4001, 3499, 3500, 3501, 2499,2500,2501]})
print(df['Rent'].mean())
one_hot_df = pd.get_dummies(df, prefix=['city'])
print(df)
print(one_hot_df)

model = linear_model.LinearRegression()
model.fit(one_hot_df[['city_NYC', 'city_SF', 'city_Seattle']],
	     one_hot_df[['Rent']])
print(model.coef_)