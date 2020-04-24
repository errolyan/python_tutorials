# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ：
   Description :  higher entropy: higher uncertainty.
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/24  13:43
-------------------------------------------------
   Change Activity:
                  2019/12/24  13:43:
-------------------------------------------------
'''
__author__ = 'yanerrol'
import torch
from torch.nn import functional as F

#higher entropy: higher uncertainty.
a = torch.full([4],0.25)
a1 = a*torch.log2(a)
print('a1',a1)
sum_a = -(a*torch.log2(a)).sum()
print('sum_a',sum_a)

a = torch.tensor([0.1,0.1,0.1,0.2])
a1 = a*torch.log2(a)
print('a1',a1)
sum_a = -(a*torch.log2(a)).sum()
print('sum_a',sum_a)

# numberical Stability
x = torch.randn(1,784)
w = torch.randn(10,784)

logits = x@w.t()
print('logits',logits,logits.type(),logits.size())

pred = F.softmax(logits,dim=1)
pred_log = torch.log(pred)
loss  = F.cross_entropy(logits,torch.tensor([3]))
print('loss',loss,loss.size())
loss1 = F.nll_loss(pred_log,torch.tensor([3]))
print('loss1',loss1,loss1.size())
