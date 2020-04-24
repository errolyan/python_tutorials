# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-27--18:24
Describe: matplot learning
'''

# coding=utf-8
import pylab as pl
import numpy as np
import mpl_toolkits.mplot3d
rho, theta = np.mgrid[0:1:40j,0:2*np.pi:40j]
c = rho**2
a = rho*np.cos(theta)
b = rho*np.sin(theta)
ax = pl.subplot(111, projection='3d')
ax.set_title('Yan Errol learning matplot');
ax.plot_surface(a,b,c)
ax.plot_surface(a,b,c,rstride=2, cstride=1)
#设置坐标轴标签
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('C')
pl.show()
