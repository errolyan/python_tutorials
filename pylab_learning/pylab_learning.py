# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-08-06  14:25
'''

from pylab import *
X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)
plot(X,C)
plot(X,S)
savefig("exercice_2.png",dpi=72)
show()
