# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  
@Email:2681506@gmail.com   
@Wechat:qq260187357
@Date:  2019-05-18  13:40
@File：ditu.py
@Describe:地图
'''
print(__doc__)

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.figure(figsize=(16,8))
m = Basemap()
m.drawcoastlines()
plt.show()
