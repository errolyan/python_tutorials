# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-11--16:30
Describe:
'''
import time


class Single_Cyc_Node(object):
    """单链表的结点"""

    def __init__(self, item):
        # _item存放数据元素
        self.item = item
        # _next 是指向下一个节点的标识
        self.next = None


'''
单链表的操作：
is_empty() 链表是否为空
length() 链表长度
travel() 遍历整个链表
add(item) 链表头部添加元素
append(item) 链表尾部添加元素
insert(pos, item) 指定位置添加元素
remove(item) 删除节点
search(item) 查找节点是否存在
'''


class Single_Cyc_link_list(object):
    """ 单向链表 """

    def __init__(self):
        self._head = None

    def is_empty(self):
        """ 判断是否是空链表"""
        return self._head is None

    def length(self):
        """ 链表的长度"""
        # cur 初始指向头节点
        cur = self._head
        Node_number = 0
        while cur is not None:
            Node_number += 1
            # 节点后移
            cur = cur.next
        return Node_number

    def travel(self):
        """遍历链表"""
        cur = self._head
        while cur is not None:
            print(cur.item)
            cur = cur.next

        print("")

    def add(self, item):
        """ 头部添加元素 """
        # 创建一个保存item值的节点

        node = Single_Cyc_Node(item)
        # 新节点的链接域next指向头节点，即_head指向的位置
        if self.is_empty():
            self._head = node
            node.next = self._head
        else:
            # 添加节点指向_head
            node.next = self._head
            # 移动链表尾部，将尾部节点的next指向node
            cur = self._head
            while cur.next != self._head:
                cur = cur.next
            cur.next = node
            # _head 指向添加node的
            self._head = node

    def append(self, item):
        """尾部添加元素"""
        node = Single_Cyc_Node(item)
        # 先判断链表是否为空，为空链表，则_head指向新的节点

        if self.is_empty():
            self._head = node
            node.next = self._head

        # 若不为空，则查找尾部，即尾部的next指向新的节点
        else:
            cur = self._head
            while cur.next is not None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        """ 指定的位置插入元素 """
        # 如果指定的位置是第一个元素前，则调用add
        if pos <= 0:
            self.add(item)
        # 指定的位置是尾部，则执行尾部添加
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            node = Single_Cyc_Node(item)
            Node_number = 0
            # pre用来指向指定位置pos的前一个位置pos-1，初始从头节点开始移动到指定位置
            pre = self._head
            while Node_number < (pos - 1):
                Node_number += 1
                pre = pre.next
            # 先将新节点node的next指向插入位置的节点
            node.next = pre.next
            # 将插入位置的前一个节点的next指向新节点
            pre.next = node

    def remove(self, item):
        """删除节点"""
        # 链表为空，则直接返回
        if self.is_empty():
            return
        cur = self._head
        pre = None
        # 若头节点的元素就是要查找的元素item
        if cur.item == item:
            if cur.next != self._head:
                # 先找到尾部节点的next指向第二个节点
                while cur.next != self._head:
                    cur = cur.next
                # cur指向了尾部节点
                cur.next = self._head.next
                self._head = self._head.next

            else:
                # 链表中只有一个节点
                self._head = None
        else:
            pre = self._head
            # 第一个节点不是要删除的
            while cur.next != self._head:
                # 找到了要删除的元素
                if cur.item == item:
                    # 删除
                    pre.next = cur.next
                    return
                else:
                    pre = cur
                    cur = cur.next
            # cur 指向尾节点
            if cur.item == item:
                # 尾部删除
                pre.next = cur.next

    def search(self, item):
        """链表查找节点是否存在，并返回True或者False"""
        """查找节点是否存在"""
        if self.is_empty():
            return False
        cur = self._head
        if cur.item == item:
            return True
        while cur.next != self._head:
            cur = cur.next
            if cur.item == item:
                return True
        return False


if __name__ == "__main__":
    ll = Single_Cyc_link_list()
    ll.add(1)
    ll.add(2)
    ll.append(3)
    ll.insert(2, 4)
    ll.insert(4, 5)
    ll.insert(0, 6)
    print("length:", ll.length())
    ll.travel()
    print(ll.search(3))
    print(ll.search(7))
    ll.remove(1)
    print("length:", ll.length())
    ll.travel()
