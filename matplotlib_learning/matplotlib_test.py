# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-10-05  10:27
'''

import  matplotlib.pyplot as plt
import numpy as np
#导入包
x=np.linspace(-3,3,50)#产生-3到3之间50个点
y1=2*x+1#定义函数
y2=x**2
# 绘制直线
plt.figure()
plt.plot(x,y1)
plt.show()

# num=3表示图片上方标题 变为figure3，figsize=(长，宽)设置figure大小
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
# 红色虚线直线宽度默认1.0
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
plt.show()

# num=3表示图片上方标题 变为figure3，figsize=(长，宽)设置figure大小
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
# 红色虚线直线宽度默认1.0
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')

plt.xlim((-1,2))#设置x轴范围
plt.ylim((-2,3))#设置轴y范围

#设置坐标轴含义， 注：英文直接写，中文需要后面加上fontproperties属性
plt.xlabel(u'价格',fontproperties='SimHei')
plt.ylabel(u'利润',fontproperties='SimHei')

# 设置x轴刻度
# -1到2区间，5个点，4个区间，平均分：[-1.,-0.25,0.5,1.25,2.]
new_ticks=np.linspace(-1,2,5)
print(new_ticks)
plt.xticks(new_ticks)

# 设置y轴刻度
'''
设置对应坐标用汉字或英文表示，后面的属性fontproperties表示中文可见，不乱码，
内部英文$$表示将英文括起来，r表示正则匹配，通过这个方式将其变为好看的字体
如果要显示特殊字符，比如阿尔法，则用转意符\alpha,前面的\ 表示空格转意
'''
plt.yticks([-2,-1.8,-1,1.22,3.],
           ['非常糟糕','糟糕',r'$good\ \alpha$',r'$really\ good$','超级好'],fontproperties='SimHei')
plt.show()


# num=3表示图片上方标题 变为figure3，figsize=(长，宽)设置figure大小
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y2)
# 红色虚线直线宽度默认1.0
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
# 设置边框/坐标轴
gca='get current axis/获取当前轴线'
ax=plt.gca()
# spines就是脊梁，即四个边框
# 取消右边与上边轴
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# matlibplot并没有设置默认的x轴与y轴方向，下面就开始设置默认轴
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


# 设置坐标原点
# 实现将(0,-1)设为坐标原点
# 设置y轴上-1为坐标原点的y点,把x轴放置再-1处
ax.spines['bottom'].set_position(('data',-1)) # 也可以是('axes',0.1)后面是百分比，相当于定位到10%处
# 设置x轴上0为坐标原点的x点，将y轴移置0处
ax.spines['left'].set_position(('data',0))

# 再写一遍以下代码，因为以上使用set_position后，中文会显示不出来

plt.yticks([-2,-1.8,-1,1.22,3.],
           ['非常糟糕','糟糕',r'$good\ \alpha$',r'$really\ good$','超级好'],fontproperties='SimHei')
plt.show()
