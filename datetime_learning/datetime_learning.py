# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import datetime

a = datetime.date.today()
print(a)
print(a.year,a.month, a.day, a.min)
print(a.__getattribute__('year'),a.__getattribute__('month'),a.__getattribute__('day'))
'''
__eq__(…)	等于(x==y)	x.__eq__(y)
__ge__(…)	大于等于(x>=y)	x.__ge__(y)
__gt__(…)	大于(x>y)	x.__gt__(y)
__le__(…)	小于等于(x<=y)	x.__le__(y)
__lt__(…)	小于(x	x.__lt__(y)
__ne__(…)	不等于(x!=y)	x.__ne__(y)
'''
b = datetime.date(2019,11,6)
print(a.__eq__(b))
print(a.__ge__(b))
print(a.__gt__(b))
print(a.__le__(b))
print(a.__lt__(b))
print(a.__ne__(b))
# 获取两个日期差
print(a.__sub__(b).days)
print(a.__rsub__(b))

# ISO 标准化日期
print('isocalendar',a.isocalendar())
print('isoformat',a.isoformat())

a = datetime.datetime.now()
print('a',a)
print('date',a.date())
'''
%y	两位数的年份表示（00-99）
%Y	四位数的年份表示（000-9999）
%m	月份（01-12）
%d	月内中的一天（0-31）
%H	24小时制小时数（0-23）
%I	12小时制小时数（01-12）
%M	分钟数（00=59）
%S	秒（00-59）
%a	本地简化星期名称
%A	本地完整星期名称
%b	本地简化的月份名称
%B	本地完整的月份名称
%c	本地相应的日期表示和时间表示
%j	年内的一天（001-366）
%p	本地A.M.或P.M.的等价符
%U	一年中的星期数（00-53）星期天为星期的开始
%w	星期（0-6），星期天为星期的开始
%W	一年中的星期数（00-53）星期一为星期的开始
%x	本地相应的日期表示
%X	本地相应的时间表示
%Z	当前时区的名称
%%	%号本身
'''
# 获得本周一至今天的时间段并获得上周对应同一时间段
today = datetime.date.today()
this_monday = today - datetime.timedelta(today.isoweekday() - 1)
last_monday = this_monday - datetime.timedelta(7)
last_weekday = today -datetime.timedelta(7)
print('last_weekday',last_weekday)
print("last_monday", last_monday)
print("this_monday", this_monday)

