# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

# 下载数据集
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root = './data',train = True, download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,shuffle = True, num_workers =2)

testset = torchvision.datasets.CIFAR10(root = './data',train = False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 4,shuffle = True, num_workers =2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 可视化数据集
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 +0.5 # 非规范化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(" ".join('%5s'%classes[labels[j]] for j in range(4)))

# 定义卷积网络
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 模型加载到cpu或gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device',device)
net.to(device)

# 定义优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

# 训练网络
for epoch in range(2):
    running_loss = 0.0
    for i,data in enumerate(trainloader,0):
        # data= [inputs,labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # 初始化优化器
        optimizer.zero_grad()

        # 前向 + 后向 + 优化器
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 2000 ==1999:
            print('[%d,%5d] loss:%.3f'%(epoch + 1,i+1,running_loss /2000))
            running_loss = 0.0

print('Finished Training')

# 保存权重
PATH_checkpoint = './data/cifar_net.pth'
torch.save(net.state_dict(),PATH_checkpoint)

# 加载模型
net_predict = Net()
net_predict.load_state_dict(torch.load(PATH_checkpoint))


# 验证测试集
# dataiter = iter(testloader)
# images, labels = dataiter.next()
# outputs = net_predict(images)
# _,predicted = torch.max(outputs,1 )
# print("Predicted"," ".join(' '.join('%5s' % classes[labels[j]] for j in range(4))))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net_predict(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


