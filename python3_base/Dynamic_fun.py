# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import types

# 定义一个类
class Person(object):
    num = 0
    def __init__(self,name,age):
        self.name = name
        self.age = age

    def eat_fun(self):
        print("eat food")

# 定义一个实例方法
def run(self,speed):
    print("%s在跑，速度是%d"%(self.name,speed))
@classmethod
def testClass(cls):
    cls.num =20

@staticmethod
def teststaticmethod():
    print("__static method__")

p = Person("lao wang",24)
# 调用在class方法
p.eat_fun()

# 添加实例方法
p.run = types.MethodType(run,p)
p.run(190)

# 类添加类方法
Person.testClass = testClass
print(Person.num)
Person.testClass()
print(Person.num)

# 类绑定静态方法
Person.teststaticmethod = teststaticmethod
Person.teststaticmethod()
p.del(age)