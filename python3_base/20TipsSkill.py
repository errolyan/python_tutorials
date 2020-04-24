# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： 20TipsSkill
   Description :  AIM: 20
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/12  16:00
-------------------------------------------------
   Change Activity:
          2020/3/12 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

'''tip 1 字符串反转'''
str1 = "Newland"
print("conversion",str1[::-1])
print(str1[0:3:1],str1[-7:],str1[::1]==str1[::])

from functools import reduce
print(reduce(lambda x,y:y+x,str1))


'''判断是否是回文'''
str2 = "abcba"
str3 = "abcda"

def fun(string):
    if string ==string[::-1]:
        print("该字段是回文字段")
        return True
    else:
        print("该字段不是回文字段")
        return False

print(fun(str3))
print(fun(str2))

'''拼接'''
list1 = ["I","love","python"]
print(" ".join(list1))

import re
str4 = "I , love , python"
print(" ".join(re.split('\W+',str4)))

'''将元素重复'''
str4 = "python"
list2 = [1,2,3]
print(str4 * 2)
print(list2 * 2)

'''将列表展开'''
# from iteration_utilities import deepflatten
# list2 = [[12,5],[4],[6,7,8][[1,2],3,[[2,3]]]]
# print(list(deepflatten(list2)))
list2 = [[12,5],[4],[6,7,8],[1,2],3,]
def flatten(lst):
    res = []
    for i in lst:
        if isinstance(i,list):
            res.extend(flatten(i))
        else:
            res.append(i)
    return res

print(flatten(list2))

from collections import Counter

list3 = ['P',"p","O","1"]
count = Counter(list3)
print(count['P'])


'''合并两个字典'''
dict1 = {"a":1,"b":2}
dict2 = {"c":3,"d":4}
dict1.update(dict2)
print(dict1)

