# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-19  23:54
'''
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device（“cuda：0”）＃取消注释以在GPU上运行

# N是批量大小; D_in是输入维度;
# H是隐藏的维度; D_out是输出维度。
N, D_in, H, D_out = 64, 1000, 100, 10

# 创建随机Tensors以保持输入和输出。
# 设置requires_grad = False表示我们不需要计算渐变
# 在向后传球期间对于这些Tensors。
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 为权重创建随机Tensors。
# 设置requires_grad = True表示我们想要计算渐变
# 在向后传球期间尊重这些张贴。
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 前向传播：使用tensors上的操作计算预测值y;
      # 由于w1和w2有requires_grad=True，涉及这些张量的操作将让PyTorch构建计算图，
    # 从而允许自动计算梯度。由于我们不再手工实现反向传播，所以不需要保留中间值的引用。
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 使用Tensors上的操作计算和打印丢失。
    # loss是一个形状为()的张量
    # loss.item() 得到这个张量对应的python数值
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用autograd计算反向传播。这个调用将计算loss对所有requires_grad=True的tensor的梯度。
    # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
    loss.backward()

    # 使用梯度下降更新权重。对于这一步，我们只想对w1和w2的值进行原地改变；不想为更新阶段构建计算图，
    # 所以我们使用torch.no_grad()上下文管理器防止PyTorch为更新构建计算图
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 反向传播后手动将梯度设置为零
        w1.grad.zero_()
        w2.grad.zero_()


import numpy as np
import torch
a_arrary = np.array([2,3,3])
b_tensfor = torch.from_numpy(a_arrary)
print('toch',b_tensfor)
print(type(b_tensfor),b_tensfor.type())
rst = isinstance(b_tensfor,torch.LongTensor)
print('rst',rst)

# 随机生成张量 dim2
aa = torch.randn(3,3)
print('aa',aa,type(aa))

# dim 3
bb = torch.rand(1,2,3)
print('bb',bb,type(bb))

# dim 4
aa = torch.rand(2,3,4,4)
print('aa',aa,"\naa.type()",aa.type(),'\ntype(aa)',type(aa),'\naa.size()',aa.size())
print('列举多少个数字',aa.numel())
print('列举维度',aa.dim(),'\n')

a = torch.tensor(2.2)
print('a.shape',a.shape)
print('torch.Size([])',torch.Size([]))
print('len(a.shape)',len(a.shape),'\n',a.size())

# ones()
a = torch.ones(2)
print(a.shape)
print(torch.Size([2]))

# fome list
cc = torch.tensor([[2,3.2],[1.0,0.89]])
print('cc',cc,cc.type())

# empty 并初始化
dd = torch.empty(1)
print(dd,dd.type())

# Tensor
dd = torch.Tensor(2,3)
print(dd,dd.type())

dd2 = torch.IntTensor(2,3)
print('dd2',dd2,dd2.type())

# 修改默认张量类型
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2,2]).type())

## full
ee = torch.full([2,3],7)
print('ee',ee,ee.type())

# arange
ff = torch.arange(0,10)
print('ff',ff,ff.type())

ff2 = torch.arange(0,10,2)
print('ff2',ff2,ff2.type())

# linspace / logspace
gg = torch.linspace(0,10,steps=3)
print(gg,gg.type())