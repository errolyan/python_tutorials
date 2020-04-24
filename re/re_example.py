# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： re_example
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/1/1  16:40
-------------------------------------------------
   Change Activity:
                  2020/1/1  16:40:
-------------------------------------------------
'''

__author__ = 'yanerrol'
import re
# 查找python
s = 'i love python very much'
pat = 'python'
r = re.search(pat,s)
print(r.span())

# 查找1
s = '山东省潍坊市第1中学高三1班'
pat = '1'
r = re.finditer(pat,s)
for i in r:
    print(i)

# 搜索数据
s =  '一共20行代码运行时间13.59s'
pat = r'\d+\.?\d*'
r = re.findall(pat,s)
print('r3333',r)

#^匹配字符串的开头
s ='This module provides regular expression matching operations similar to those found in Perl'
pat =r'^[emrt]'
r = re.findall(pat, s)
print(r)

#re.I 忽略大小写
s ='This module provides regular expression matching operations similar to those found in Perl'
pat = r'^[emrt]'
r = re.compile(pat, re.I).search(s)
print(r)

#使用正则提取单词
s ='This module provides regular expression matching operations similar to those found in Perl'
pat =r'\s[a-zA-Z]+'
r = re.findall(pat, s)
print(r)

# 只捕获单词，去掉空格
s ='This module provides regular expression matching operations similar to those found in Perl'
pat ='\s([a-zA-Z]+)'
r = re.findall(pat, s)
print(r)
