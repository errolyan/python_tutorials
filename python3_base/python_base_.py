# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-08-03  16:33
'''

import os
tuple1 = (1,2,"hello")
print(tuple1.count(1))
print(tuple1.index(2))

# set
set1 = {1,2,33,4,5,66,7,8,99,0}
set2 = {1,2,33,4,5,66,9999,7777,666,555}
set1.add(333)
s = set1.difference(set2)
print(set1)
print(s)
import random

print(random.random())
print(random.seed(2))

lis = range(0,10,2)
print(sum(lis))

result = any(a%2 for a in range(0,10,2))
print('result',result)

result = all(a%2 for a in range(0,10,2))
print('result',result)

lis2 = list(enumerate(lis))
print("lis2",lis2)

lis3 = zip(lis,lis2)
print(lis3)

a = {"a":1,"b":2}
b = {"b":1,"c":33}
a.update(b)
print("a",a)

# 链方式比较
i = 3
print('1 < i <= 3',1 < i <= 3)

# 求解字符串的字节长度
def str_byte_len(mystr):
    return (len(mystr.encode('utf-8')))

print(str_byte_len('hello world'))

# 寻找第n次出现的位置
def search_n(s,c,n):
    size = 0
    for i,x in enumerate(s):
        if x==c:
            size += 1
        if size == n:
            return i
    return -1

print(search_n('fdasadffard','a',3))

# 去掉最高最低求平均
def score_mean(lst):
    lst.sort()
    lst2= lst[1:(len(lst)-1)]
    return round(sum(lst2)/len(lst2),2)

score_mean([11,0,2,4,5,8,3])