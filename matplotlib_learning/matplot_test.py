# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-10-12  17:24
'''
import  matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-3,3,50)#产生-3到3之间50个点

y1=2*x+1#定义函数
y2=x**2
# 绘制直线
plt.figure()
plt.plot(x,y1)
# num=3表示图片上方标题 变为figure3，figsize=(长，宽)设置figure大小
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
# 红色虚线直线宽度默认1.0
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
plt.show()