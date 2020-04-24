# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： Deriatives_math
   Description :  求导数
   Envs        :  torch
   Author      :  yanerrol
   Date        ： 2019/12/10  23:17
-------------------------------------------------
   Change Activity:
                  2019/12/10  23:17:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import torch
import numpy as np


def func(x):
    return (x**3 - 7*x**2 + 11*x)

x = torch.tensor(2.0, requires_grad=True)

y = func(x)
y.backward()
print(x.grad)

import torch
import numpy as np
c = torch.Tensor([i for i in range(5)])
print('c',c[0],c.shape,type(c),)
