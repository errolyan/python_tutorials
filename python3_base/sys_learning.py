# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-14  14:03
'''
import sys
variable = 30
print(sys.getsizeof(variable)) # 24

s = "programming is awesome"
print(s.title())
print(s.upper())
print(s.lower())

a = {"a":1,"b1":2}
b = {"b":3,"v":6}
a.update(b) # 两个字典合并，如果有重复的键值对，则以后面的为主
print(a)






