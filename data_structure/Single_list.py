# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-11--00:25
Describe: 单向链表是最简单的一种形式，它每个结点包含两个域，一个信息域，一个事链接域，链接指向表中的下一个一个节点，而最后一个节点的链接域指向一个空值。
'''


import time


class SingleNode(object):
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


class Single_link_list(object):
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

        node = SingleNode(item)
        # 新节点的链接域next指向头节点，即_head指向的位置
        node.next = self._head
        self._head = node

    def append(self, item):
        """尾部添加元素"""
        node = SingleNode(item)
        # 先判断链表是否为空，为空链表，则_head指向新的节点

        if self.is_empty():
            self._head = node

        # 若不为空，则查找尾部，即尾部的next指向新的节点
        else:
            cur = self._head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    def insert(self,pos,item):
        """ 指定的位置插入元素 """
        # 如果指定的位置是第一个元素前，则调用add
        if pos <= 0:
            self.add(item)
        # 指定的位置是尾部，则执行尾部添加
        elif pos > (self.length()-1):
            self.append(item)
        else:
            node = SingleNode(item)
            Node_number =0
            # pre用来指向指定位置pos的前一个位置pos-1，初始从头节点开始移动到指定位置
            pre = self._head
            while Node_number < (pos-1):
                Node_number += 1
                pre = pre.next
            # 先将新节点node的next指向插入位置的节点
            node.next = pre.next
            # 将插入位置的前一个节点的next指向新节点
            pre.next = node


    def remove(self, item):
        """删除节点"""
        cur = self._head
        pre = None
        while cur != None:
            # 找到了指定元素
            if cur.item == item:
                # 如果第一个就是删除的节点
                if not pre:
                    # 将头指针指向头节点的后一个节点
                    self._head = cur.next
                else:
                    # 将删除位置前一个节点的next指向删除位置的后一个节点
                    pre.next = cur.next
                break
            else:
                # 继续按链表后移节点
                pre = cur
                cur = cur.next


    def search(self, item):
        """链表查找节点是否存在，并返回True或者False"""
        cur = self._head
        while cur != None:
            if cur.item == item:
                return True
            cur = cur.next
        return False

if __name__ == "__main__":
    ll = Single_link_list()
    ll.add(1)
    ll.add(2)
    ll.append(3)
    ll.insert(2, 4)
    print("length:", ll.length())
    ll.travel()
    print(ll.search(3))
    print(ll.search(5))
    ll.remove(1)
    print("length:", ll.length())
    ll.travel()
