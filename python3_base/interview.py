#coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-29--11:14
Describe: python 面试基础题目
'''

import sys

dic = {"name":"YanErrol","age":28}

del dic["name"]

print (dic)

dic2 = {"name":"ls"}

dic.update(dic2)
print (dic)

# 字典键排序

def fun(args_f,*args_v):
    print (args_f)
    print (args_v,type(args_v))
    for x in args_v:
        print(x)

fun("a","b","c","d")

a = range(100)
print(a,type(a))


# learning map()
lis = [1,2,3,5]
def func(x):
    return x**2

result = map(func,lis)

result = [i for i in  result if i > 10]
print (result)
print (lis)


# random
import random
import numpy as np

result = random.randint(10,20)
res = np.random.randn(2)
resu = random.random()

print ("正整数",result)
print ("5个随机数",res)
print ("0-1随机小数",resu)
print (id(res))
print (id(result))
print (id(resu))
print (id(func))

#字符排序
s = "sshhkfhsfjosfjklfjdslafja"
print (s,type(s),id(s))
s = set(s)
print (s,type(s),id(s))
s = list(s)
print (s,type(s),id(s))
s.sort(reverse=False)
print (s,type(s),id(s))

# lambda 函数
sum = lambda a,b,c:a*b/c
print (sum(12,2,3),type(sum),id(sum))

from collections import Counter

a = "khkfkfjlakflkfldsklfjlfjfjj"
res = Counter(a)
print (res,type(res),id(res))