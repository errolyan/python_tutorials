# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  
@Email:2681506@gmail.com   
@Date:  2019-05-20  16:49
@File：python_class.py
@Describe:类的学习，属性访问权限的界定
'''

import os

class Person(object):

    def __init__(self,name,age):
        self._name = name
        self._age = age

    @property
    def name(self):
        return self._name

    @property
    def age(self):
        return self.age

    @age.setter
    def age(self,age):
        self._age = age

    def play(self):
        if self._age <= 16:
            print("%s正在玩飞机"%self._name)
        else:
            print("%s正在斗地主"%self.name)

def main():
    person = Person("zhangsan",10)
    person.play()
    person.age = 73
    person.play()


if __name__ == "__main__":
    main()