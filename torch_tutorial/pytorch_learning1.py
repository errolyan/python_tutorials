# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-18  15:16
'''


from __future__ import print_function
import torch
print('torch.__version__:',torch.__version__)
print('__doc__:',__doc__)
'''
构造一个矩阵
'''
x = torch.empty(5, 3)
print(x)

'''
构造一个随机矩阵
'''
x = torch.rand(5, 3)
print(x)

'''
构造一个long型数据结构
'''
x = torch.zeros(5,3,3,dtype=torch.long)
print(x)

'''
列表转tensor
'''
x = torch.tensor([4,3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
# new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)

# override dtype!
print(x,x.size())

y = torch.rand(5,3)
print('x+y',x+y)
print(torch.add(x, y))

#改变大小：如果你想改变一个 tensor 的大小或者形状，你可以使用 torch.view:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()
print(x.grad)