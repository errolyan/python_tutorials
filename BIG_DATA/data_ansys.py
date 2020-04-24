# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol  Email:2681506@gmail.com   Wechat:qq260187357
Date:2019-05-13--11:26
File：
Describe: 数据相关性分析，
'''

print(__doc__)

import pandas as pd

# Step 0 - Read the dataset, calculate column correlations and make a seaborn heatmap
data = pd.read_csv('https://raw.githubusercontent.com/drazenz/heatmap/master/autos.clean.csv')
corr = data.corr()
ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
