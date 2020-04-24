# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： learning_1
   Description :  AIM: 计算简单的数学题
                  Functions: 1. 开根号
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/2/20  23:34
-------------------------------------------------
   Change Activity:
          2020/2/20 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

from sympy import *
x = symbols("x")
a = Integral(cos(x)*exp(x),x)
print(Eq(a,a.doit()))


# 计算开根号
import sympy
print(sympy.sqrt(3))

# sympy中变量要使用之前必须定义

from sympy import symbols,expand,factor
x,y = symbols("x y")
expres = x + 2*y
print('expres = ',expres)

print(expres - x)
print(expres - y)

# 展开式
expand_expr = expand(expres*x)
print(expand_expr)
# 简写形式
print(factor(expand_expr))
x = symbols('x') # sympy表达式变量
expr = x + 1
x = 2 # python变量
print(expr)

x = symbols('x')
expres= x+1
print(x,type(x))
print(expres.subs(x,2))
print(Eq(x+1,4)) #等式

# 表达式计算
a = (x+1)**2
b = (x**2+2*x+1)
print(simplify(a-b))
b = (x**2-2*x+1)
print(simplify(a-b))

a = cos(x)**2 - sin(x)**2
b = cos(2*x)
print(a.equals(b)) # 等于，余弦函数展开式

print(True ^ False)

print('Rational(1,3)',Rational(1,3)) # 分数
print('Integer(1)/Integer(3)',Integer(1)/Integer(3)) # 分数

# 等式转换
str_expr = "x**2 + 3*x -1/2"
expr = sympify(str_expr)
print(expr)

# 表达式计算得到一个浮点数值
expr = sqrt(8)
print("result",expr.evalf())

print("pi",pi.evalf(5)) # 5是有效数字个数

# 变量代入式子并保留有效数字
expr = cos(2*x)
print("expr",expr.evalf(subs={x:0}),2)

