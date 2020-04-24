# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： re_serch-repale
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/14  14:31
-------------------------------------------------
   Change Activity:
          2020/2/14 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import re
def replace_num(str):
	numDict = {'0':'〇','1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九'}
	print(str.group())
	return numDict[str.group()]
my_str = '2018年6月7号'
a = re.sub(r'(\d)', replace_num, my_str)
print(a)

# \d 匹配数字[0-9] （整数）
s = '一共 20 行代码运行时间 0.59s'
pat = r'\d+' # + 表示匹配数字 (\d 表示数字的通用字符)1 次或多次
r = re.findall(pat,s)
print(r)

# 匹配浮点数和整数
pat = r'\d+\.?\d+' # ? 表示匹配小数点 (\.)0 次或 1 次，这种写法有个小 bug，不能 匹配到个位数的整数
r = re.findall(pat,s)
print(r)

# 匹配单词
s = 'This module provides regular expression matching operations similar to those found in Perl'
pat = r'\s([a-zA-Z]+)'
r = re.findall(pat,s)
print(r)

# 忽略大小写
s = 'That'
pat = r't'
r = re.findall(pat,s,re.I)
print(r)

# 替换匹配的字符串
content="hello 12345, hello 456321"
pat=re.compile(r'\d+') # 要替换的部分
m=pat.sub("666",content)
print(m) # hello 666, hello 666