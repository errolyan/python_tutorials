# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： plot
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == python >=3.5
                  pip install sympy  -i https://pypi.douban.com/simple
   Author      :  yanerrol
   Date        ： 2020/2/20  16:23
-------------------------------------------------
   Change Activity:
          2020/2/20 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

from sympy import Symbol,pprint,log,exp,Rational,sin,summation,sqrt,oo,Integral
from sympy.plotting import plot
x = Symbol('x')
y = Symbol('y')
i = Integral(log((sin(x)**2+1)*sqrt(x**2+1)),(x,0,y))
print("i",i)
i.evalf(subs={y:1})
plot(i,(y, 1, 5))

def sumns():
    s = summation(1/x**y,(x,1,oo))
    print('s',s)
    plot(s,(y,2,10))

def Finite_sums():
    p = plot(summation(1 / x, (x, 1, y)), (y, 2, 10))
    p[0].only_integers = True
    p[0].steps = True
    p.show()

if __name__ == '__main__':
    sumns()
    Finite_sums()




