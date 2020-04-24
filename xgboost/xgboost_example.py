# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  xgboost
@Evn     :  pip install xgboost
@Date    :  2019-07-13  17:53
'''

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# data process
data_train = pd.read_csv("../data/波士顿房价数据集/train.csv")
data_test = pd.read_csv("../data/波士顿房价数据集/test.csv")

#删除不相关属性
X = data_train.drop(['ID', 'medv'], axis=1)
y = data_train.medv

#将数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1, max_depth=8,
alpha = 8, n_estimators=500, reg_lambda=1)
# train
xg_reg.fit(X_train, y_train)
# 预测
x_test = data_test.drop(['ID'], axis=1)
predictions = xg_reg.predict(x_test)
ID = (data_test.ID).astype(int)
result = np.c_[ID, predictions]

np.savetxt("../data/波士顿房价数据集/" + 'xgb_submission.csv', result, fmt="%d,%.4f",header='ID,medv', delimiter=',', comments='')