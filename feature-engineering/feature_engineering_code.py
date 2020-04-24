# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
# 导入一些常用包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('fivethirtyeight')

#解决中文显示问题，Mac

from matplotlib.font_manager import FontProperties

plt.rcParams['figure.figsize'] = (10.0, 4.0) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
plt.rcParams['font.family'] = ['Arial Unicode MS'] #正常显示中文

# 引入第 1 个数据集 Salary_Ranges_by_Job_Classification
salary_ranges = pd.read_csv('./data/Salary_Ranges_by_Job_Classification.csv')
print(salary_ranges.head())
# 绘制条形图
salary_ranges['Grade'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')
plt.show()
plt.savefig("./pictures/test.png")

# 绘制饼图
salary_ranges['Grade'].value_counts().sort_values(ascending=False).head(5).plot(kind='pie')
plt.show()
plt.savefig("./pictures/bingtu.png")

# 绘制箱体图
salary_ranges['Union Code'].value_counts().sort_values(ascending=False).head(5).plot(kind='box')
plt.show()
plt.savefig("./pictures/xiangti.png")

# 引入第 2 个数据集
climate = pd.read_csv('./data/GlobalLandTemperaturesByCity.csv')
print(climate.head())
print(climate.info())

# 查看字段的基本统计情况（只会显示数值型变量）
print(climate.describe())

# 移除缺失值
climate_nona = climate.dropna(axis=0, inplace=True)
print(climate_nona)

# 检查缺失个数
isnull_num = climate.isnull().sum()
print(isnull_num)

# 枚举所有变量值的数量
nunique_num = climate['AverageTemperature'].nunique()
print(nunique_num)

# 绘制直方图
climate['AverageTemperature'].hist()
plt.show()
plt.savefig("./pictures/zhifang.png")

# 日期转换, 将dt 转换为日期，取年份, 注意map的用法
climate['dt'] = pd.to_datetime(climate['dt'])
climate['year'] = climate['dt'].map(lambda value: value.year)

# 只看中国
climate_sub_china = climate.loc[climate['Country'] == 'China']
climate_sub_china['Century'] = climate_sub_china['year'].map(lambda x:int(x/100 +1))
print(climate_sub_china.head())

# 为每个世纪（Century）绘制平均温度的直方图
climate_sub_china['AverageTemperature'].hist(by=climate_sub_china['Century'],
                                            sharex=True,
                                            sharey=True,
                                            figsize=(10, 10),
                                            bins=20)
plt.show()
plt.savefig("./pictures/zhifang11.png")

# 按世纪来分组计算温度的均值
climate_sub_china.groupby('Century')['AverageTemperature'].mean().plot(kind='line')
plt.show()
plt.savefig("./pictures/quxian.png")

# 绘制散点图
x = climate_sub_china['year']
y = climate_sub_china['AverageTemperature']

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(x, y)
plt.show()
plt.savefig("./pictures/zhifang11sandian.png")

# 引入第 3 个数据集(皮马印第安人糖尿病预测数据集)
pima_columns = ['times_pregment','plasma_glucose_concentration','diastolic_blood_pressure','triceps_thickness',
                'serum_insulin','bmi','pedigree_function','age','onset_disbetes']

pima = pd.read_csv('./data/pima.data', names=pima_columns)
print(pima.head())