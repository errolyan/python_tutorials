# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： Tips_python2
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/30  23:49
-------------------------------------------------
   Change Activity:
          2020/3/30 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

# 四舍五入
print(round(10.22245,3))

# 排序算法
a = [1,4,5,6,7]
print(sorted(a,reverse=True))

# 求和函数
a = [1,23]
print(sum(a))

# 求平均数
print('平均数',sum(a)/len(a))

# 元组：不可改变的序列类型
print(tuple(a),type(tuple(a)))

# 展开多层的list嵌套
a = [[1,3],[2,4],[3,5],["abc","def"]]
a1 = [y for x in a for y in x]
print("a1",a1)

# 求众数
def top_num (lst):
    return max (lst,default='列表为空',key=lambda v:lst.count(v))

lst = [1,2,3,1,1,3]
print(top_num(lst))

# 列表反转
r = lst[::-1]
print(r)

# map实现向量运算
lst1 = [1,2,3,4,5]
lst2 = [3,4,5,6,7]
print(list(map(lambda x,y:x*y-1,lst2,lst1)))

# 生成逆袭序列
print(list(range(10,-1,-1)))

print('=========================字符串和正则=======================================')
# 反转字符串
string = 'learning python'
print(''.join(reversed(string)))
print(string[::-1])

import re
s = 'i love python very much'
pat = 'python'
r = re.search(pat,s)
print(r.span()) #(7,13)

# 查找一个字符的索引
s = '山东省潍坊市青州第 1 中学高三 1 班'
pat = '1'
r = re.finditer(pat,s)
for i in r:
    print(i)

