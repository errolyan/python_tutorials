# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： series_learning
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/2/20  23:24
-------------------------------------------------
   Change Activity:
          2020/2/20 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

from sympy import Symbol, cos, sin, pprint


def main():
    x = Symbol('x')

    e = 1/cos(x)
    print('')
    print("Series for sec(x):")
    print('')
    pprint(e.series(x, 0, 10))
    print("\n")

    e = 1/sin(x)
    print("Series for csc(x):")
    print('')
    pprint(e.series(x, 0, 4))
    print('')

if __name__ == "__main__":
    main()



import sympy
from sympy import Mul, Pow, S


def main():
    x = Pow(2, 50, evaluate=False)
    y = Pow(10, -50, evaluate=False)
    # A large, unevaluated expression
    m = Mul(x, y, evaluate=False)
    # Evaluating the expression
    e = S(2)**50/S(10)**50
    print("%s == %s" % (m, e))

if __name__ == "__main__":
    main()

