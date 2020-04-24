# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-11  09:02
'''

# 导入turtle包的所有内容:
from turtle import *
# '''
# 绘制矩形
# '''
# # 设置笔刷宽度:
# width(4)
#
# # 前进:
# forward(200)
# # 右转90度:
# right(90)
#
# # 笔刷颜色:
# pencolor('red')
# forward(100)
# right(90)
#
# pencolor('green')
# forward(200)
# right(90)
#
# pencolor('blue')
# forward(100)
# right(90)

'''
绘制五角星
'''
def drawStar(x, y):
    pu()
    goto(x, y)
    pd()
    # set heading: 0
    seth(0)
    for i in range(10):
        fd(40)
        rt(144)

for x in range(0, 250, 50):
    drawStar(x, 0)

done()

