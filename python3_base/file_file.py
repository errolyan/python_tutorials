# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-08-03  12:35
'''

import os
# 删除文件
if os.path.exists("test.txt"):
    os.remove("test.txt")
else:
    print("the file does not exit")

# 文件夹删除
import os
if os.path.exists("test"):
    os.rmdir("test")

# 绝对值
print(abs(-45))

# 可迭代
a = [1,12,3]
b = [2,3]
c= 1
print(all(a))
print(any(a))
print(zip(a,b)) # 压缩两个对象为一个

# 字符操作
d = "aaDDDavv"
print(d.capitalize()) # 手写字母大写
print(d.casefold()) # 大写字母转化为小写
print(d.center(1)) # 返回中间的字母
print(d.count("a")) # 统计数目
print(d.encode()) # 编码放肆
print(d.endswith("v"))
print(d.expandtabs())
print("find",d.find("a"))
print(d.isalnum())
print(d.isalpha())
print(d.isdecimal())
print(d.isdigit())
print(d.isidentifier())
print(d.islower())
print(d.isnumeric())
print(d.isprintable())
print(d.isspace())
print(d.istitle())
print(d.isupper())
print(d.split("DDD"))

# 列表操作
a = [1,33,44,67,0,"qq"]
a.append("hello")
a.remove(1)
b = a.copy()
a.insert(22,5)
print(a)
print(b)
print(b.index(0))
# print(b.sort()) #必须是同一个类型

'''
Dic
'''
dic1 = {"name":"yanerle","age":"30","work":"it"}
lis1 = dic1.keys()
lis2 = dic1.values()
# dic3 = dic1.copy()
# dic4 = dic1.get("name")
# dic5 = dic1.items()
# dit2 = dic1.clear()
dic1.update()
print(lis1,lis2)
print(dic1)



















