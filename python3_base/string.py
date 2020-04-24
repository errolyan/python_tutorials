# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol
@Email:2681506@gmail.com
@Wechat:qq260187357
@Date:  2019-05-17  23:57
@File：string.py
@Describe:字符串的操作
'''
print(__doc__)

import os
import time
str1 = "hello world"
# 首字母大写
print(str1.capitalize())
# 字母大写
print(str1.upper())
print(str1.find("ll"))
print(str1.find("yan"))
print(str1.index("or"))
print(str1.startswith("He"))
print(str1[2::2])
print(str1[1::2])
print(str1.isdigit())
print(str1.isalpha())
print(str1.isalnum())

content = '北京欢迎你为你开天辟地…………'
while True:
    # 清理屏幕上的输出
    os.system('clear')  # os.system('clear')
    print(content)
    # 休眠200毫秒
    time.sleep(0.2)
    content = content[1:] + content[0]