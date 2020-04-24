# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''

class Test():
    pass

print("class",Test.__name__)
print("Class Test", Test())
print(help(Test))


age = 23
name = "Yan"
def main():
    pass

class new():
    pass

newob = new()

print(age.__class__,age.__class__.__class__)
print(name.__class__,name.__class__.__class__)
print(new.__class__,new.__class__.__class__)
print(newob.__class__,newob.__class__.__class__)

import types
class Person():
    def __init__(self,name = None, age = None):
        self.name = name
        self.age = age

    def eat(self):
        print("chi fan")

def run(self,speed):
    print("%s在移动，速度是%d"%(self.name,speed))


# 定义一个类方法
@classmethod
def testClass(cls):
    cls.num =100

# 定义一个静态方法
@staticmethod
def testStatic():
    print("---static method----")

p  = Person("老王", 24)
print(p)
p.sex = "nan"
print(p.sex)
p.eat()
p.run = types.MethodType(run,p)
p.run(100)
#给Person类绑定类方法
Person.testClass = testClass
#调用类方法
print(Person.num)
Person.testClass()
print(Person.num)


#给Person类绑定静态方法
Person.testStatic = testStatic

#调用静态方法
Person.testStatic()

