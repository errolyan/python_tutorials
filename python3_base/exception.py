# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-29--21:59
Describe:异常诊断
'''

import time

def func():
    try:
        for i in range(5):
            if i >3:
                raise Exception("数字大于3了==")

    except Exception as ret:
        print (ret)


func()

import re

a = "张明 99分"
ret = re.sub(r"\d+","100",a)
print (ret)

a = [1,2,3]
b = [4,5,6]
print(a+b)