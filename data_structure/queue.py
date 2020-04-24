# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-11--18:06
Describe:
'''
import time


class Queue(object):
    def __init__(self):
        """queue def"""
        self.items = []

    def is_empty(self):
        """ 判断是否为空 """
        return self.items == []

    def in_queue(self, item):
        """ 加入元素 """
        return self.items.insert(0, item)

    def out_queue(self):
        """ 取出元素 """
        return self.items.pop()

    def peek(self):
        """ 取出栈的顶元素 """
        return self.items[len(self.items) - 1]

    def size(self):
        """ 返回栈的大小 """
        return len(self.items)


def main():
    queue = Queue()
    queue.in_queue("Hello")
    queue.in_queue("World")
    queue.in_queue("Errol")
    print(queue,type(queue))
    print(queue.peek())
    print(queue.size())
    print(queue.out_queue())
    print(queue.out_queue())
    print(queue.out_queue())


if __name__ == "__main__":
    main()
