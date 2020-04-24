# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import os
import torch
import numpy as np
from torchvision.datasets import mnist
from torch  import nn
from torch.autograd import Variable

# 数据标准化处理
def data_std(x):
    x = np.array(x,dtype='float32')/255
    x = (x-0.5)/0.5 #标准化
    x = x.reshape((-1,))# 拉平
    x  = torch.from_numpy(x)
    return x

# 数据准备
train_set = mnist.MNIST('./data',train=True,transform=data_std,download= True)
test_set = mnist.MNIST('./data',train=False,transform=data_std,download= True)
print('train_set',train_set,type(train_set))
print('test_set',test_set,type(test_set))
a,a_label = train_set[0] # a是训练数据第一个的像素点，a_label为训练数据第一个的样本的标签

from torch.utils.data import DataLoader
# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器，也就是将数据进行排序标号，shuffle也就是打乱数据
train_data = DataLoader(train_set, batch_size=64, shuffle=True)#训练数据
test_data = DataLoader(test_set, batch_size=128, shuffle=False)#测试数据
a, a_label = next(iter(train_data))
print('a',a,type(a),a.shape)
print('a_label',a_label,type(a_label))
net = nn.Sequential(nn.Linear(784,400),nn.ReLU(),nn.Linear(400,200),nn.ReLU(),nn.Linear(200,100),nn.ReLU(),nn.Linear(100,10))

# net
# 定义loss函数
criterion = nn.CrossEntropyLoss()# 定义交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr =0.1) # 使用随机下降，学习率是0.1
# 开始训练，定义空列表进而存储后边训练得到的数据
losses = []
acces = []
eval_losses =[]
eval_acces = []

for e in  range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for im,label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = criterion(out,label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _,pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        # torch中训练需要将其封装即Variable，此处封装像素即784
        label = Variable(label)
        # 此处为标签
        out = net(im)
        # 经网络输出的结果
        loss = criterion(out, label)
        # 得到误差
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        # 得到出现最大值的位置，也就是预测得到的数即0—9
        num_correct = (pred == label).sum().item()
        # 判断是否预测正确
        acc = num_correct / im.shape[0]
        # 计算准确率
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))
import matplotlib.pyplot as plt
plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.plot(np.arange(len(acces)), acces)
plt.title('train acc')
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')