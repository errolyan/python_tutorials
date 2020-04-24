# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： tensor_adan
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/24  08:54
-------------------------------------------------
   Change Activity:
                  2019/12/24  08:54:
-------------------------------------------------
'''
__author__ = 'yanerrol'
import torch
'''
if cond true,a;else b
'''
cond = torch.tensor([[0.6,0.9],[0.4,0.6]])
a = torch.tensor([[0,0],[0,0]])
b = torch.tensor([[1,1],[1,1]])
print(torch.where(cond>0.5,a,b))

# activation functions :sigmoid
a = torch.linspace(-100,100,10)
print('a',a)
b = torch.sigmoid(a)
print('b',b,b.size())

# tanh
a = torch.linspace(-1,1,10)
print('a',a,a.size())
b = torch.tanh(a)
print('b',b,b.size())

# F.relu

from torch.nn import functional as F
a  = torch.linspace(-1,1,10)
b = torch.relu(a)
print('b',b,b.size())
c = F.relu(a)
print('c',c,c.size())

# 损失函数：均方误差（Mean Squared Error)MSE
# 损失函数：交叉熵 (CEL,Cross Entropy Loss)

x = torch.ones(1)
w = torch.full([1],2,requires_grad=True)
mse = F.mse_loss(torch.ones(1),x*w)
print('mse',mse,mse.size())
b = torch.autograd.grad(mse,[w])

print('b',b)




