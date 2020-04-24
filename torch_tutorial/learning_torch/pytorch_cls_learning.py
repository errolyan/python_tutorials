# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： pytorch_cls_learning
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/4/16  09:26
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

class ManualLinearRegression(nn.Module):

    def __init__(self,):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self,x):
        # Computes the outputs / predictions
        return self.a + self.b * x

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

torch.manual_seed(42)
# Now we can create a model and send it at once to the device
model = ManualLinearRegression().to(device)
# We can also inspect its parameters using its state_dict
print(model.state_dict())
lr = 1e-1
n_epochs = 1000
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)
for epoch in range(n_epochs):
 # What is this?!?
 model.train()
 # No more manual prediction!
 # yhat = a + b * x_tensor
 yhat = model(x_train_tensor)
 loss = loss_fn(y_train_tensor, yhat)
 loss.backward()
 optimizer.step()
 optimizer.zero_grad()
print(model.state_dict())