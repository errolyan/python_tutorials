# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  积分
@Evn     :  
@Date    :   - 
'''
# 单积分
import scipy.integrate
from numpy import exp
f = lambda x:exp(-x**2)
i = scipy.integrate.quad(f,0,1)
print('i',i)
# 多重积分

import scipy.integrate
from numpy import exp
from numpy import sqrt

f = lambda x,y:16*x*y
g = lambda x: 0
h = lambda y:sqrt(1-4*y**2)
i = scipy.integrate.dblquad(f, 0, 0.5,g,h)
print('i',i)

# scipy 差值
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
x = np.linspace(0,4,12)
y = np.cos(x**2/3+4)
print('x,y',x,y)
plt.plot(x,y,'o')
plt.show()

from  scipy import interpolate
## 一维差值
f1 = interpolate.interp1d(x,y,kind ='linear')
f2 = interpolate.interp1d(x,y,kind = "cubic")
xnew = np.linspace(0,4,30)
plt.plot(x,y,'o',xnew,f1(xnew),'-',xnew,f2(xnew),'--')
plt.legend(['data','linear','cubic','nearest'],loc='best')
plt.show()

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 *np.random.randn(50)
plt.plot(x,y,'ro',ms =5)
plt.show()

spl = UnivariateSpline(x,y)
xs = np.linspace(-3,3,1000)

spl.set_smoothing_factor(0.5)
plt.plot(xs,spl(xs),'b',lw = 3)
plt.plot(xs,spl(xs),'g',lw = 3)
plt.show()

