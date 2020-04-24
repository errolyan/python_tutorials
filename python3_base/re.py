# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-29--14:59
Describe: 正则化
'''

import re

string = "not 999 , . FFFound 张三 00 福州"
lis = string.split(" ")
print (lis)
res = re.findall ('\d+|,.',string) # 依次匹配数字字符
print (res)
res1 = re.findall ('\d+',string) # 依次匹配数字
print (res1)
res2 = re.findall ('\d+|[a-zA-Z]+|[,.]+',string) # 依次匹配数字
print (res2)

for i in res2:
    if i in lis:
        print (i)
        lis.remove(i)

new_str = " ".join(lis)
print (lis)
print (new_str)

