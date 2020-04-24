# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： Myself_torch.py
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/4/16  08:43
-------------------------------------------------
   Change Activity:
          2020/4/16 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

device = 'cuda' if torch.cuda.is_available()  else 'cpu'
print(device)

# Data generator
np.random.seed(42)
x = np.random.rand(100,1)
y = 1 + 2*x + .1 * np.random.rand(100,1)
print('x',x,'\ny',y)
# shuffle the indices
idx = np.arange(100)
print('idx',idx)

# Uses first 80 random indices for train
train_idx = idx[:80]

# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)
# Here we can see the difference - notice that .type() is more useful
# since it also tells us WHERE the tensor is (device)
print(type(x_train), type(x_train_tensor), x_train_tensor.type())

lr = 1e-1
n_epochs = 1000
torch.manual_seed(42)
a = torch.randn(1,requires_grad=True,dtype=torch.float,device=device)
b = torch.randn(1,requires_grad=True,dtype=torch.float,device=device)
for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor
    error = y_train_tensor - yhat
    loss = (error ** 2).mean()
    # we just tell toch to work its way backwards from the specified loss!
    loss.backward()
    print(a.grad)
    print(b.grad)
    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad

    # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
    a.grad.zero_()
    b.grad.zero_()
print(a, b)
