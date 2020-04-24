# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： example3
   Description :  画出堆积图来看占比关系
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/6  00:21
-------------------------------------------------
   Change Activity:
                  2020/2/6  00:21:
-------------------------------------------------
'''
__author__ = 'yanerrol'

# 画出堆积图来看占比关系
# 导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

# 导入泰坦尼的数据集
data_train = pd.read_csv("./data/titanic/Train.csv")
data_train.head()

# 设置figure_size尺寸
plt.rcParams['figure.figsize'] = (5.0, 4.0)

#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.8)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
plt.show()