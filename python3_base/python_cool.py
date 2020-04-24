# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
fun_sq = lambda x:x*x
print(fun_sq(5))

nums = [1/3, 333/7, 2323/2230, 40/34, 2/3]
nums_squared_1 = map(fun_sq, nums)
print(list(nums_squared_1))
from functools import reduce
product = reduce(lambda x, y: x * y, nums)
print(product)