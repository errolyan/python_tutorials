# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''


# 检查列表是否有重复元素
def all_unique(lst):
    return len(lst) == len(set(lst))

x = [1,1,2,3,4]
y = [1,2,3]
print(all_unique(x),all_unique(y))


# 变量使用内存情况
import sys

variable = 300
print(sys.getsizeof(variable))

# 字节占用
def byte_size(string):
    return (len(string.encode('utf-8')))

print(byte_size("hello world"))

# 大写第一个字母
print("hello".title())

# 使用filter函数 去掉bool类型

def compact(lst):
    return list(filter(bool,lst))
print(compact([0, 1, False, 2,  3,  'a' ,  's' , 34]))

# 列表的差
def difference(a,b):
    set_a = set(a)
    set_b = set(b)
    comparison = set_a.difference(set_b)
    return comparison

print(difference([1,2,3],[1,2,5,6]))