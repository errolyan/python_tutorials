# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  是否迭代
@Evn     :  
@Date    :  2019-09-09  18:42
'''
__author__ = "Yan Errol"
from functools import reduce
from collections import Iterable,Iterator
var1 = 123
var2 = "123"
var3 = [1,2,3]
var4 = {}
var5 = ()
if isinstance(var1,Iterable):
    print(var1)
else:
    print(var1,"is not iterable")

if isinstance(var2,Iterable):
    print(var2)

if isinstance(var3,Iterable):
    print(var3)

if isinstance(var4,Iterable):
    print(var4)

if isinstance(var5,Iterable):
    print(var5)

print([x*x for x in range(10)])

g = (x * x for x in range(10))
print(next(g),'-----')

for i in range(9):
    print(next(g))

print(isinstance((x for x in range(10)), Iterator))
print(isinstance([],Iterator))
print(isinstance({},Iterator))

def func(x):
    return x*x

r =  map(func,[1,2,3,4,5])
print(list(r))

print(list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9])))
def add(x, y):
    return x + y

a = reduce(add, [1, 3, 5, 7, 9])
print(a)


class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def get_grade(self):
        if self.score >= 90:
            return 'A'
        elif self.score >= 60:
            return 'B'
        else:
            return 'C'

s1 = Student("yan",100)
print(s1.get_grade())
print(dir(s1))
print(type(s1))
print(isinstance(s1,Student))
s2 = Student("ya",100)
s2.name = "Errol"
print(s2.name)

def set_age(self,age):
    self.age = age

# 给类添加新的方法
from types import MethodType
s2.set_age = MethodType(set_age,s2)

# 调用新的方法
s2.set_age(25)
print(s2.age)