# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol  Email:2681506@gmail.com   Wechat:qq260187357
Date:2019-05-14--21:02
File：**kwargs
Describe:learning **kwargs
'''
print(__doc__)

from functools import wraps

def logit(fun):
    @wraps(func)
    def with_logging(*args,**kwargs):
        print(func.__name__ + "was called")
        return func(*args,**kwargs)
    return with_logging

@logit
def add_func(x):
    return x + x

result = add_func(8)





