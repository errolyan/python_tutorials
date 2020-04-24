# -*- coding:utf-8 -*-
# /usr/bin/python

__author__ = 'yanerrol'

import pandas as pd
import numpy as np

print(pd.__version__)

array = [1,2,3,4]
df = pd.DataFrame(array)
print('df :\n',df)

# 数组转序列
df_series = pd.Series(array)
print('df_series:\n',df_series)

# 字典转序列
d = {'a':1,'b':2,'c':3,'d':4,'e':5}
df = pd.Series(d)
print('df :\n',df)

# numpy 创建序列
dates = pd.date_range('today',periods=6) # 定义时间序列作为 index
num_arr = np.random.randn(6,4) # 传入 numpy 随机数组
columns = ['A','B','C','D'] # 将列表作为列名
df = pd.DataFrame(num_arr, index = dates, columns = columns)
print(df)

# 从表格读取序列
# df = pd.read_csv('test.csv', encoding='gbk, sep=';')


# 字典创建df 并设置索引
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data, index=labels)
print(df)

# 显示df 的基础信息
print(df.info(),df.describe())


# 展示前三行
print('展示前三行:\n',df.head(3),'\n',df.loc['a':'c'])

# 取出某几列
print(df.loc[:,['animal','age']])

# 取出345行和某几列
print(df.loc[df.index[[3, 4, 8]], ['animal', 'age']])

# 取出age大于3的行
print(df[df['age']>3])

# 取出有缺失值的行
print(df[df['age'].isnull()])

# 取出age在2，4的行
print(df[(df['age']>2) & (df['age']>4)],df[df['age'].between(2, 4)])


# 改动某一行的值
df.loc['f', 'age'] = 1.5
print(df)

# 在df中插入新行k，然后删除该行
#插入
df.loc['k'] = [5.5, 'dog', 'no', 2]
# 删除
df = df.drop('k')
print(df)

# 统计df中的种类数
print(df['animal'].value_counts())


# 删除重复的行
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
print(df)
df1 = df.loc[df['A'].shift() != df['A']]
# 方法二
# df1 = df.drop_duplicates(subset='A')
print(df1)

# 一个群数值的df,每一个数字减去改行的平均数
df = pd.DataFrame(np.random.random(size=(5, 3)))
print(df)
df1 = df.sub(df.mean(axis=1), axis=0)
print(df1)

## 一个有5列的DataFrame，求哪一列的和最小
df = pd.DataFrame(np.random.random(size=(5, 5)), columns=list('abcde'))
print(df)
print(df.sum().idxmin())

### 可视化操作
import matplotlib.pyplot as plt
df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})
plt.style.use('ggplot')
df.plot.scatter("xs", "ys", color = "black", marker = "x")
plt.show()

df = pd.DataFrame({"productivity":[5,2,3,1,4,5,6,7,8,3,4,8,9],
                   "hours_in"    :[1,9,6,5,3,9,2,9,1,7,4,2,2],
                   "happiness"   :[2,1,3,2,3,1,2,3,1,2,2,1,3],
                   "caffienated" :[0,0,1,1,0,0,0,0,1,1,0,1,0]})

df.plot.scatter("hours_in", "productivity", s = df.happiness * 100, c = df.caffienated)
plt.show()