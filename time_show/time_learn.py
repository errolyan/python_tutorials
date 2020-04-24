# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： time_learn
   Description :  AIM: 时间显示
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install datetime -i https://pypi.douban.com/simple
   Author      :  yanerrol
   Date        ： 2020/3/2  16:48
-------------------------------------------------
   Change Activity:
          2020/3/2 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import calendar
from datetime import date
mytime = date.today()
years_calendar = calendar.calendar(2020)
print(years_calendar)


