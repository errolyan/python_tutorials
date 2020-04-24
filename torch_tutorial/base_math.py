# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： base_math
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/23  22:50
-------------------------------------------------
   Change Activity:
                  2019/12/23  22:50:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import torch
a = torch.rand(3,4)
b = torch.rand(4)
# 每行对应的元素都增加
c = a+b
print('\na',a,'\nb',b,'\nc',c)
# 每行对应的元素相加
c = torch.add(a,b)
print(c)

c = a-b-(torch.sub(a,b))
print(c)

# 除
c = a/b-torch.div(a,b)
print(c)

# #
# # # a = torch.tensor([[3,3],[3,3]])
# # # b = torch.ones(2,2)
# # # b = torch.ones([2,2])
# # # print(b,b)
# # # c = a@b
# # # print(c)

# 指数
a = torch.full([2,2],3)
b = a.pow(2)
print(b,b.size())
b = a**2
print(b,b.size())

a = b.rsqrt()
print('a',a.size())

# 对数函数
a = torch.exp(torch.ones(2,2))
b = torch.log(a)
print(b,b.size())

# floor ceil
a = torch.tensor(3.14)
print(a.floor(),a.ceil(),a.trunc(),a.frac())
print(a.round())

# 正则化
a = torch.full([8],1)
b = a.view(2,4)
c = a.view(2,2,2)
print(a,'\b',b,'\n',c)
print(a.norm(1),'\n',b.norm(1),'\n',c.norm(1))
print(a.norm(2),'\n',b.norm(2),'\n',c.norm(2))

# 依据维度正则
print(b.norm(1,dim=1))

# 求解平均值 求和 最值
a = torch.arange(8).view(2,4).float()
print(a.min(),a.max(),a.mean(),a.prod(),a.sum())

# 判断大小
print(torch.equal(a,a))


