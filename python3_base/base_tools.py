# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  检查对象的内存使用情况
@Evn     :  
@Date    :  2019-09-29  14:04
'''

#检查对象的内存使用情况
import sys
variable = 30
print(sys.getsizeof(variable))


# 测试字符的字节数
def byte_size(string):
    return(len(string.encode('utf-8')))
print(byte_size('？'))
# 4
print(byte_size('Hello World')) # 11


# 重复打印字符N次
n= 2;
s="Programming";
print(s * n);

# 合并两个词典
def merge_two_dicts(a, b):
    c = a.copy() # make a copy of a
    c.update(b) # modify keys and values of a with the ones from b
    return c
a = { 'x': 1, 'y':2}
b = { 'y' : 3, 'z' : 4}
print(merge_two_dicts(a, b))

import sys
print(sys.path)

# 二进制
print(bin(10)) # 10进制转化为二进制

print(int('1010',2))# 二进制转化为10进制

print(hex(10)) # 10进制转16进制
# 10进制转16进制
print(hex(10))

# 16进制转2进制
bin(0xa)

import functools
print(dir(functools))
