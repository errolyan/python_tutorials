# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： Perceptron
   Description :  torch 感知器
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/24  10:04
-------------------------------------------------
   Change Activity:
                  2019/12/24  10:04:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import torch
from torch.nn import functional as F

def input_data():
    '''
    输入数据
    '''
    return torch.randn(1,10)

def weight_init():
    '''
    权重初始化
    '''
    return torch.randn(1,10,requires_grad=True)

def activate_func(x,w):
    '''
    激活函数
    '''
    res = torch.sigmoid(x@w.t())
    return res

def loss_func(x,w,y):
    '''
    定义损失函数
    '''
    res = activate_func(x,w)
    loss = F.mse_loss(y,res)
    return loss
def op_fun(x):
    optimizer = torch.optim.Adam([x], lr=1e-3)
    return optimizer
def backward_fun(x,loss,w):
    '''
    后向传播
    '''
    optimizer = op_fun(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    w.grad

    return w

def main():
    x = input_data()
    w = weight_init()
    y = torch.ones(1,1)
    for i in range(3):
        loss = loss_func(x,w,y)
        print('loss',loss)
        w = backward_fun(x,loss,w)
        print('w',w)

if __name__=='__main__':
    main()
