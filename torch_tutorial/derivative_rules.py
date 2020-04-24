# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： derivative_rules
   Description :  求导法则
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/24  10:35
-------------------------------------------------
   Change Activity:
                  2019/12/24  10:35:
-------------------------------------------------
'''
__author__ = 'yanerrol'
import torch
from torch.nn import functional as F

x = torch.tensor(1.)
w1 = torch.tensor(2.,requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2.,requires_grad=True)
b2 = torch.tensor(1.)

y1 = x*w1+b1
y2 = y1*w2+b2

dy2_dy1 = torch.autograd.grad(y2,[y1],retain_graph=True)[0]
dy1_dw1 = torch.autograd.grad(y1,[w1],retain_graph=True)[0]
dy2_dw1 = torch.autograd.grad(y2,[w1],retain_graph=True)[0]
w3 = dy2_dy1*dy1_dw1
print('w3',w3,w3.type())
w3 = dy2_dw1
print('w3',w3,w3.type())
