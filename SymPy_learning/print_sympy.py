# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： print_sympy
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/2/21  09:21
-------------------------------------------------
   Change Activity:
          2020/2/21 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

from sympy import *
x,y,z = symbols("x y z")
init_printing()
print(Integral(sqrt(1/x),x)) # 积分式子显示

