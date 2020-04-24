# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： Myself_learning
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/4/15  21:34
-------------------------------------------------
   Change Activity:
          2020/4/15 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import numpy as np
import matplotlib.pyplot as plt

# Sets learning rate
lr = 1e-1
# Defines number of epochs
n_epochs = 1000


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

print('x_train',x_train,'\n  y_train \n', y_train)
print('x_val',x_val,'\n  y_val \n', y_val)

plt.figure()
plt.subplot(1,2,1)
plt.scatter(x_train, y_train)
plt.title('Train datasets')
plt.subplot(1,2,2)
plt.scatter(x_val, y_val)
plt.title('Val datasets')

#plt.show()

# Initializes parameters "a" and "b" randomly
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)
print(a, b)

# training
for epoch in range(n_epochs):
    # computes our model's predictions output
    yhat = a + b*x_train
    # How wrong is our model? that's the error
    error = (y_train - yhat)
    # It is a regression,so it computes mean squared error(MSE)
    loss = (error**2).mean()
    print('loss = ',loss)
    # 计算梯度
    a_grad = -2*error.mean()
    b_grad = -2*x_train*error.mean()
    # 更新权重
    a = a - lr*a_grad
    b = b - lr*b_grad
print('a',a,'b',b)

from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])

# pytorch
import torch
import torch.optim as optim
import torch.nn as nn
#from torchviz import make_dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device



