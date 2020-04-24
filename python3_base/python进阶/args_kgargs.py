# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： args_kgargs
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/25  10:11
-------------------------------------------------
   Change Activity:
                  2019/12/25  10:11:
-------------------------------------------------
'''
__author__ = 'yanerrol'

# *argv 非键值对的参数，个数不限制
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    print(argv,type(argv))
    for arg in argv:
        print("another arg through *argv:", arg)
test_var_args('yasoob', 'python', 'eggs', 'test')

def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))
greet_me(name="yasoob")

# map
def print_func(i):
    i=i**2
    return i
lis = [1,2,3]
lis = list(map(print_func,lis))
print(lis)

from functools import reduce
product = reduce( (lambda x,y: x * 2), [1, 2, 3, 4] )
print(product,type(product))

condition = True
print(2 if condition else 1/0)

# 装饰器
def hi(name="yaoob"):
    def greet():
        return "now you are in the greet() function"
    def welcome():
        return "now you are in the welcome() function"
    if name == "yasoob":
        return greet
    else:
        return welcome
a = hi()
print(a())

''' 双向序列 '''
from collections import deque
d = deque()
d.append(1)
d.append(2)
d.append(3)
print('d',d,len(d),type(d))
d = deque(range(5))
print(len(d))
print(d.popleft())
print(d.pop())
print(d)

''' try except '''
try:
    file = open('test.txt', 'rb')
except IOError as e:
    print('An IOError occurred. {}'.format(e.args[-1]))
finally:
    print("This would be printed whether or not an exception")

''' lambda '''
add = lambda x,y:x+y
print(add(3,5))
print(add(4,4))
print(add(5,4))



