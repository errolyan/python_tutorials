# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： sympy_base
   Description :  AIM :  python求解数学题
                  Functions: 1. 创建字符式子并打印出阿拉伯式子
                             2. 
   Envs        :  python == 3.5
                  pip install sympy -i https://pypi.douban.com/simple
   Author      :  yanerrol
   Date        ： 2020/2/20  15:53
-------------------------------------------------
   Change Activity:
          2020/2/20 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

from sympy import Symbol,pprint,log,exp,Rational,sin,limit,sqrt,oo

def show():
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    e = (a*b*b + 2**b*a)*c
    print('')
    pprint(e)
    print('')

def differentiation():
    a = Symbol('a')
    b = Symbol('b')
    e = (a + 2 * b) ** 5

    print("\nExpression : ")
    print()
    pprint(e)
    print("\n\nDifferentiating w.r.t. a:")
    print()
    pprint(e.diff(a))
    print("\n\nDifferentiating w.r.t. b:")
    print()
    pprint(e.diff(b))
    print("\n\nSecond derivative of the above result w.r.t. a:")
    print()
    pprint(e.diff(b).diff(a, 2))
    print("\n\nExpanding the above result:")
    print()
    pprint(e.expand().diff(b).diff(a, 2))
    print()

def define_functions():
    a = Symbol('a')
    b = Symbol('b')
    e = log((a + b) ** 5)
    print("define_functions")
    pprint(e)
    print('\n')

    e = exp(e)
    pprint(e)
    print('\n')

    e = log(exp((a + b) ** 5))
    pprint(e)
    print

def limit_functions():
    '''
    求极限
    :return:
    '''

    def sqrt3(x):
        return x ** Rational(1, 3)

    def show(computed, correct):
        print("computed:", computed, "correct:", correct)

    x = Symbol("x")

    show(limit(sqrt(x ** 2 - 5 * x + 6) - x, x, oo), -Rational(5) / 2)

    show(limit(x * (sqrt(x ** 2 + 1) - x), x, oo), Rational(1) / 2)

    show(limit(x - sqrt3(x ** 3 - 1), x, oo), Rational(0))

    show(limit(log(1 + exp(x)) / x, x, -oo), Rational(0))

    show(limit(log(1 + exp(x)) / x, x, oo), Rational(1))

    show(limit(sin(3 * x) / x, x, 0), Rational(3))

    show(limit(sin(5 * x) / sin(2 * x), x, 0), Rational(5) / 2)

    show(limit(((x - 1) / (x + 1)) ** x, x, oo), exp(-2))

if __name__ == '__main__':
    show()
    differentiation()
    define_functions()
    limit_functions()



