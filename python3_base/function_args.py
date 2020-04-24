# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： function_args
   Description :  函数参数
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/4  21:22
-------------------------------------------------
   Change Activity:
                  2020/2/4  21:22:
-------------------------------------------------
'''
__author__ = 'yanerrol'

def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')


def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))

greet_me(name="yasoob")
greet_me(age="14")
kwargs = {'name':'zhangsan','age':15}
greet_me(**kwargs)


# 生成器也是一种迭代器，主要是为了解决一次性加载内存数据过大
def generator_function():
    for i in range(10):
        yield i

for item in generator_function():
    print(item)

# Map会将一个函数映射到一个输入列表的所有元素上
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
print(squared)

# filter过滤列表中的元素，并且返回一个由所有符合要求的元素所构成的列表，符合要求即函数映射到该元素时返回值为True.
number_list = range(-5, 5)
less_than_zero = filter(lambda x: x < 0, number_list)
print(list(less_than_zero))

# reduce
from functools import reduce
product = reduce( (lambda x, y: x * y), [1, 2, 3, 4] )
print('product',product)


# 用装饰器 做日志
from functools import wraps

def logit(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return with_logging

@logit
def addition_func(x):
   """Do some math."""
   return x + x

result = addition_func(4)
print(result)

# 集合推导式
squared = {x**2 for x in [1, 1, 2]}
print(squared)

# 异常 异常处理是一种艺术
try:
    file = open('test.txt', 'rb')
except IOError as e:
    print('An IOError occurred. {}'.format(e.args[-1]))

try:
    file = open('test.txt', 'rb')
except EOFError as e:
    print("An EOF error occurred.")
    raise e
except IOError as e:
    print("An error occurred.")
    raise e


a = [(1, 2), (4, 1), (9, 10), (13, -3)]
a.sort(key=lambda x: x[1])
print(a)

'''
Python代码中来调用C编写的函数-ctypes，SWIG，Python/C API。每种方式也都有各自的利弊。
首先，我们要明确为什么要在Python中调用C？
    常见原因如下：
        你要提升代码的运行速度，而且你知道C要比Python快50倍以上
        C语言中有很多传统类库，而且有些正是你想要的，但你又不想用Python去重写它们
        想对从内存到文件接口这样的底层资源进行访问
        不需要理由，就是想这样做
'''