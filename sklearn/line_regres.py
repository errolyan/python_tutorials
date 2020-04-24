# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''

import matplotlib.pyplot as plt
import numpy as np

a = [[1,2,3,4],[2,3,4,5],[3,4,5,6],]
b = [2,2,2,2]
c = np.multiply(a,b)
print(c,type(c))

b1 = [[2],[2],[3],[4]]

c = np.dot(a,b1)
print(c,type(c))

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.externals import joblib
import pandas as pd
import numpy as np


def mylinear():
    """
    线性回归直接预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()
    print(type(lb.data),type(lb.target))
    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    print(y_train, y_test)

    # 进行标准化处理(?) 目标值处理？
    # 特征值和目标值是都必须进行标准化处理, 实例化两个标准化API
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    print('type(y_train)',type(y_train))

    y_train = std_y.fit_transform(y_train.reshape(-1,1))
    y_test = std_y.transform(y_test.reshape(-1,1))

    # 训练
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    print('type(x_train),type(y_train)',type(x_train),type(y_train))
    print('lr.coef_',lr.coef_)

    joblib.dump(lr,'./test.pkl')
    # 预测房价结果
    model = joblib.load("./test.pkl")
    y_predict = std_y.inverse_transform(model.predict(x_test))
    print("保存的模型预测的结果：", y_predict)
    print("正规方程的均方误差:",mean_squared_error(std_y.inverse_transform(y_test),y_predict))



    # 2梯度下降法
    sgd = SGDRegressor()
    sgd.fit(x_train,y_train)
    print(sgd.coef_)

    #预测房价
    y_sgd_predic = std_y.inverse_transform(sgd.predict(x_test))
    print('梯度下降测试房价预测',y_sgd_predic)
    print("梯度下降均方误差",mean_squared_error(std_y.inverse_transform(y_test),y_sgd_predic))


if __name__ =="__main__":
    mylinear()