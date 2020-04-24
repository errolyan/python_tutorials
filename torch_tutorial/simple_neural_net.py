# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： simple_neural_net
   Description :  一个简单的网络
   Envs        :  torch
   Author      :  yanerrol
   Date        ： 2019/12/10  23:05
-------------------------------------------------
   Change Activity:
                  2019/12/10  23:05:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import torch
import numpy as np

torch.manual_seed(7)
feature = torch.randn(1,3)
print('feature',feature,id(feature),type(feature),feature.shape)

# Must match the shape of the features
n_input = feature.shape[1]
# Number of hidden units
n_hidden = 5
# Number of output units (for example 1 for binary classification)
n_output = 1

W1 = torch.randn(n_input,n_hidden)
W2 = torch.randn(n_hidden,n_output)

B1 = torch.randn((1,n_hidden))
B2 = torch.randn((1,n_output))
def activation(x):
    """
    Sigmoid activation function
    """
    return 1/(1+torch.exp(-x))

print("Shape of the input features: ",feature.shape)
print("Shape of the first tensor of weights (between input and hidden layers): ",W1.shape)
print("Shape of the second tensor of weights (between hidden and output layers): ",W2.shape)
print("Shape of the bias tensor added to the hidden layer: ",B1.shape)
print("Shape of the bias tensor added to the output layer: ",B2.shape)

h1 = activation(torch.mm(feature,W1)+B1)
print("Shape of the output of the hidden layer",h1.shape)
h2 = activation(torch.mm(h1,W2)+B2)
print("Shape of the output layer",h2.shape)
print(h2)