# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： tensor_tutorial
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/23  14:20
-------------------------------------------------
   Change Activity:
                  2019/12/23  14:20:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import torch

# ones
ones_tensor = torch.ones(3,3)
print('ones_tensor:',ones_tensor,ones_tensor.type())
# 判断对象是否是张量
res_is_tensor = torch.is_tensor(ones_tensor)
print('res_is_tensor:',res_is_tensor)

res_is_storage = torch.is_storage(ones_tensor)
print('res_is_storage:',res_is_storage)

# zeros
hh_tensor = torch.zeros(3,3)
print('zeros:',hh_tensor.type())

# eye
kk_tensor = torch.eye(3,4)
print('kk_tensor:',kk_tensor.type())

# 索引
a = torch.rand(4,3,8,3)
print('a[0]:',a[0].shape,a[0].size(),a[0][0].numel())

# reshape
a = torch.rand(4,1,28,28)
print(a.shape)
b = a.view(4,28*28)
print(b.size())
c = b.view(4,28,28,1)
print(c.size())

# 增加一个维度
d = c.unsqueeze(0)
print(d.shape,d.size())

# 尾部问价一个维度
e = c.unsqueeze(-1)
print(e.size())

b = torch.rand(32)
c = b.unsqueeze(1)
d = c.unsqueeze(2)
e = d.unsqueeze(0)
print('e',e.size(),e.shape)

# size*size 对应的维度相乘
print(e.repeat(4,32,1,1).shape)

print(e.repeat(4,1,1,1).shape)

# 拼接
a = torch.rand(4,32,8)
b = torch.rand(5,32,8)
c = torch.cat([a,b],dim=0)
print(c.size(),c.shape)

# 拼接
a1 = torch.rand(4,3,16,32)
a2 = torch.rand(4,3,16,32)
a3 = torch.cat([a1,a2],dim=2)
print(a3.shape)

# 堆加
a4 = torch.stack([a1,a2],dim=2)
print(a4.shape)

# 堆加
a = torch.rand(32,8)
b = torch.rand(32,8)
c = torch.stack([a,b],dim=0)
print('c',c.shape,c.size())

'''
cat：形状不一定相同，堆加的维度可以不相同
stack：相撞必须相同才可以堆加，并且多了一个维度
'''
# 拆分
aa,bb = c.split([1,1],dim=0)
print(aa.size(),bb.size())

aa,bb = c.split(1,dim=0)
print(aa.size(),bb.size())