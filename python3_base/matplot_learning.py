# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-27--18:13
Describe: matplotlib learning
'''

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
a,b = np.mgrid[-2:2:20j,-2:2:20j]
#测试数据
c=a*np.exp(-a**2-b**2)
#三维图形
ax = plt.subplot(111, projection='3d')
ax.set_title('Yan Errol learning matplot');
ax.plot_surface(a,b,c,rstride=2, cstride=1, cmap=plt.cm.Spectral)
#设置坐标轴标签
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('C')
plt.show()


