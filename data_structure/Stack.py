# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-11--17:46
Describe:
'''
import time


class Stack(object):
    """ 栈 """

    def __init__(self):
        self.items = []

    def is_empty(self):
        """ 判断是否为空 """
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        """ 弹出元素 """
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items) - 1]

    def size(self):
        """ 返回栈的大小 """
        return len(self.items)


if __name__ == "__main__":
    stack = Stack()
    stack.push("Hello World.")
    stack.push("Learing dataStruture")
    stack.push("Ok")
    print(stack.size())
    print(stack,type(stack))
    print(stack.peek())
    print(stack.pop())
    print(stack.pop())
    print(stack.pop())
