# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-11--23:11
Describe: Dequeue 两端进两端出
'''
import os,sys,re

class Deque(object):
    def __init__(self):
        """ Dequeue """
        self.items = []

    def is_empty(self):
        """ 单端是否识空双端队列 """
        return self.items ==[]

    def add_front(self,item):
        """ 前端加元素"""
        return self.items.insert(0,item)

    def del_front(self):
        """ 前端删除 """
        return self.items.pop(0)
    def add_back(self,item):
        """ 尾部添加 """
        return self.items.append(item)

    def del_back(self):
        return self.items.pop()

    def size(self):
        """ 返回大小 """
        return len(self.items)
def main():
    deque =Deque()
    deque.add_front("hello")
    deque.add_back("hello")
    deque.add_front("world")
    deque.add_back("world")
    print(deque,type(deque))
    print(deque.size())
    print("后端删除",deque.del_back())
    print("前端删除", deque.del_front())
    print("前端删除",deque.del_front())
    print("前端删除",deque.del_front())

if __name__ == "__main__":
    main()