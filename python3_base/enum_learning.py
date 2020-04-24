# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  枚举类
@Evn     :  
@Date    :  2019-09-10  09:06
'''
from enum import Enum
Month = Enum('Month',('Jan','Feb','Mar','Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
print(Month)

from datetime import datetime

now = datetime.now()
print(now)
print(now.timestamp())

from time import ctime
now1 = ctime()
print(now1)


x={1,2,'3,4'}
y={'3,4',5,6}
x = set(x)
y = set(y)
z = x&y
print(z,type(z))


from collections import namedtuple
'''
用来创建不变的元组，并且可以限制元素的个数
'''
Point = namedtuple('point',['x','y'])
p = Point(1,2)
print(p.x)
print(p.y)
print(isinstance(p,Point))



