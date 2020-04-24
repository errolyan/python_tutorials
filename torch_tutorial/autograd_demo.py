# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： autograd_demo
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/22  13:08
-------------------------------------------------
   Change Activity:
                  2019/12/22  13:08:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import  torch
from    torch import autograd


x = torch.tensor(1.)
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
c = torch.tensor(3., requires_grad=True)

y = a**2 * x + b * x + c

print('before:', a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print('after :', grads[0], grads[1], grads[2])


x = torch.ones(2,2,requires_grad=True)
print('x',x)
y = x+2
print('y',y)