# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： softmax_example
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/24  09:56
-------------------------------------------------
   Change Activity:
                  2019/12/24  09:56:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import torch
from torch.nn import functional as F

a = torch.rand(3,requires_grad= False)
a.requires_grad_()
print('a',a)

p = F.softmax(a,dim=0)
a1 = torch.autograd.grad(p[1],[a],retain_graph=True)
print('a1',a1)
a2 = torch.autograd.grad(p[2],[a])
print(a2)
