# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： class_jc
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/1  23:52
-------------------------------------------------
   Change Activity:
                  2020/2/1  23:52:
-------------------------------------------------
'''
__author__ = 'yanerrol'
class A (object):
    def show(self):
        print('base show')

class B(A):
    def show(self):
        print('derived show')

obj = B()
obj.show()

# 调用父类的方法

obj.__class__ = A
obj.show()

# 2. 方法对象
class AA (object):
    def __init__(self,a,b):
        self.__a = a
        self.__b = b
    def myprint(self):
        print('a=', self.__a, 'b=', self.__b)

    def __call__(self, num):
        print('call:', num + self.__a)

a1= AA (10,20)
a1.myprint()

a1(80)