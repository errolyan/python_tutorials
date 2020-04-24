# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  
@Email:2681506@gmail.com   
@Date:  2019-05-28  23:07
@File：python_gener.py
@Describe:迭代器和生成器
'''

'''
概念
迭代器：是访问数据集合内元素的一种方式，一般用来遍历数据，但是他不能像列表一样使用下标来获取数据，也就是说迭代器是不能返回的。
1. Iterator：迭代器对象，必须要实现next魔法函数
2. Iterable：可迭代对象，继承Iterator，必须要实现iter魔法函数
'''
from collections import Iterable,Iterator
def iterator_fun():
    '''
    迭代对象和可迭代的
    :return:
    '''
    a = [1,2,3,4]
    print(isinstance(a,Iterator))
    print(isinstance(a,Iterable))

iterator_fun()


def mkiterator_fun():
    '''
    可迭代的变成可迭代对象
    :return:
    '''
    from collections import Iterable, Iterator
    a = [1, 2, 3]
    a = iter(a)
    print(isinstance(a, Iterator))
    print(isinstance(a, Iterable))

    print(next(a))
    print("====")
    for x in a:
        print("----")
        print(x)

mkiterator_fun()