# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  装饰器
@Evn     :  
@Date    :  2019-09-09  15:54
'''
import time

def log(function_name):
    def wrapper(*args,**kw):
        print('call %s():'% function_name.__name__)
        return function_name(*args,**kw)
    return wrapper
@log
def time_now():
    print(time.ctime())

now = time_now
now()

print('now.__name__:',now.__name__)
print('time_now.__name__:',time_now.__name__)