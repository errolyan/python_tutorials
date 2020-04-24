# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： example
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/5  23:36
-------------------------------------------------
   Change Activity:
                  2020/2/5  23:36:
-------------------------------------------------
'''
__author__ = 'yanerrol'
# tips 1 无量纲化处理

from sklearn.datasets import load_iris
#导入IRIS数据集
iris = load_iris()

#标准化，返回值为标准化后的数据
from sklearn.preprocessing import StandardScaler
StandardScaler().fit_transform(iris.data)

#区间缩放，返回值为缩放到[0, 1]区间的数据
from sklearn.preprocessing import MinMaxScaler
MinMaxScaler().fit_transform(iris.data)

#归一化，返回值为归一化后的数据
from sklearn.preprocessing import Normalizer
Normalizer().fit_transform(iris.data)


# tips2 进行多项式or对数的数据交换，一个特征在当前分布下无明显的区分度，但是在一个小小的变化可能达到意想不到的
# 的效果
from sklearn.datasets import load_iris
#导入IRIS数据集
iris = load_iris()
data1 = iris.data[0]
print(data1)

'''
对数变换
这个操作就是直接进行一个对数转换，改变原先的数据分布，而可以达到的作用主要有:

1）取完对数之后可以缩小数据的绝对数值，方便计算；

2）取完对数之后可以把乘法计算转换为加法计算；

3）还有就是分布改变带来的意想不到的效果。

numpy库里就有好几类对数转换的方法，可以通过from numpy import xxx 进行导入使用。

log：计算自然对数

log10：底为10的log

log2：底为2的log

log1p：底为e的log
'''
from sklearn.datasets import load_iris
#导入IRIS数据集
iris = load_iris()

#多项式转换
#参数degree为度，默认值为2
from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures().fit_transform(iris.data)

#对数变换
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
#自定义转换函数为对数函数的数据变换
#第一个参数是单变元函数
FunctionTransformer(log1p).fit_transform(iris.data)

# python 中常用的统计图
# 导入一些常用包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#matplotlib inline
plt.style.use('fivethirtyeight')

#解决中文显示问题，Mac
# matplotlib inline
from matplotlib.font_manager import FontProperties


# 引入第 1 个数据集 Salary_Ranges_by_Job_Classification
salary_ranges = pd.read_csv('./data/Salary_Ranges_by_Job_Classification.csv')

# 引入第 2 个数据集 GlobalLandTemperaturesByCity
climate = pd.read_csv('./data/GlobalLandTemperaturesByCity.csv')
# 移除缺失值
climate.dropna(axis=0, inplace=True)
# 只看中国
# 日期转换, 将dt 转换为日期，取年份, 注意map的用法
climate['dt'] = pd.to_datetime(climate['dt'])
climate['year'] = climate['dt'].map(lambda value: value.year)
climate_sub_china = climate.loc[climate['Country'] == 'China']
climate_sub_china['Century'] = climate_sub_china['year'].map(lambda x:int(x/100 +1))

# 设置显示的尺寸
plt.rcParams['figure.figsize'] = (4.0, 4.0) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文

# 绘制条形图
salary_ranges['Grade'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
# 绘制饼图
salary_ranges['Grade'].value_counts().sort_values(ascending=False).head(5).plot(kind='pie')
# 绘制箱体图
salary_ranges['Union Code'].value_counts().sort_values(ascending=False).head(5).plot(kind='box')
# 绘制直方图
climate['AverageTemperature'].hist()
# 绘制散点图
x = climate_sub_china['year']
y = climate_sub_china['AverageTemperature']
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(x, y)
plt.show()