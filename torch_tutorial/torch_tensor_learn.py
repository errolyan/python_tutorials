# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import torch
import numpy as np

# 判断是否是张量
print(torch.is_tensor([[1,2],[2,3]]))
# 判断张量的元素个数
a = torch.tensor(2)
print(torch.numel(a))

tensor_a = torch.randn(1,2,3)
print(torch.numel(tensor_a))
tensor_b = torch.zeros(4,5)
print(tensor_b,type(tensor_b),type(type(tensor_b)))

# 创建对角为1，其他为0的2维张量
eyes = torch.eye(4)
print(eyes)

# numpy的array转化为torch的tensor
numpy_arr = np.array([[1,2],[2,3]])
print('numpy_arr',numpy_arr,type(numpy_arr),type(type(numpy_arr)))
arr_tensor = torch.from_numpy(numpy_arr)
print("arr_tensor",arr_tensor,type(arr_tensor),type(type(numpy_arr)))
arr_tensor[(1,-1)]=5
print(arr_tensor)

# 在区间start 和end上均匀间隔steps 个点
linspace_tensor = torch.linspace(0,10,steps = 100,)
print(linspace_tensor,type(linspace_tensor),type(type(linspace_tensor)))
print('torch.Size()',linspace_tensor.size())

# 在区间10^start 10^end上以对数刻度均匀间隔的steps个点。输出1维张量的长度为steps
logs_tensor = torch.logspace(-10,10,steps = 10)
print("logs_tensor",logs_tensor,type(logs_tensor))

# 全1 张量
ones_tensor = torch.ones((3,3))
print("ones_tensor",ones_tensor,type(ones_tensor),ones_tensor.size())

# 全0 tensor张量
zeros_tensor = torch.zeros(2,3)
print('zeros_tensor',zeros_tensor,zeros_tensor.size())

# 0到1内的随机数张量
rand_tensor = torch.randn(4)
print('rand_tensor',rand_tensor,type(rand_tensor))
rand_tensor = torch.randn([4,4])
print('rand_tensor',rand_tensor,type(rand_tensor))

# 固定间隔取值,左闭右开
arange_rand = torch.arange(1,4,0.2)
print('arange_rand',arange_rand,type(arange_rand),arange_rand.size())

# 张量前后串联
x = torch.randn([2,3])
y = torch.cat((x,x,x,x),0)# 按行拼接
print("y",y.size(),y)

y = torch.cat((x,x,x,x),1)# 按列拼接
print("y",y.size(),y)

# 拆分
z = torch.split(y,(1,1),dim =0)
print("z",z,type(z))

# 转至
y1 = torch.t(y)
print("y1", y1, type(y1),y1.size())
#
# # 创建正太分布的张量
# std_tensor = torch.normal(mean=torch.arange(1, 6))
# print('std_tensor',std_tensor,std_tensor.size())

# 保存对象
torch.save(z,f = './z.pkl')
# 加载对象
zz = torch.load(f = './z.pkl')
print('zz',zz,type(zz))

# 计算输入向量的绝对值
y2 = torch.abs(y1)
print('y2',y2,type(y2),y2.size())

x1 = torch.abs(torch.FloatTensor([-1.1,2.1,-6,-8]))
print(x1,type(x1),x1.size())

# 求每个张量值的反余弦值
a = torch.rand(4)
print("a",a)
a = torch.acos(a)
print("a反余弦函数",a)

# 求每个张量值的余弦值
a = torch.rand(4)
print("a",a)
a = torch.cos(a)
print("a余弦函数",a)

# 求每个张量值的双曲余弦值
a = torch.rand(4)
print("a",a)
a = torch.cosh(a)
print("a双曲余弦函数",a)

# 求每个张量元素值的反正弦值
a = torch.rand(4)
print("a",a)
a = torch.asin(a)
print("a反正弦函数",a)

# 求每个张量元素值的反正切函数
a = torch.rand(4)
print("a",a)
a = torch.atan(a)
print("a反正切",a)

# 张量加固定的标量值
b = torch.add(a,30)
print('b',b,b.size())

# 除法 张量的每个值除以标量值
zz = torch.div(b,3)
print("除法",zz,zz.size())

# 除法 张量的每个值除以另一个张量的每个值
zz = torch.div(b,b)
print("除法两个维度相同的张量相处",zz,zz.size())


# 复杂数学计算：tensor + (tensor1/tensor2)*value
aa = torch.addcdiv(a,2,a,b)
print('aa',aa,aa.size())

# 复杂计算：tensor + （tensor1*tensor2）*value
bb = torch.addcmul(a,2,a,b)
print('bb',bb,bb.size())

# 天井函数，对张量的每个元素向上取整
a = torch.rand(4)
print("a",a)
a = torch.ceil(a)
print("天井函数向上取整",a)

# 幂函数，以张量的每个元素作为底数，
a = torch.rand(4)
bb = torch.pow(a,4)
print('幂函数',bb,bb.size())

# 指数对每个张量的元素取以e为底的指数
exp_one = torch.exp(ones_tensor)
print("指数",exp_one,exp_one.size())

# 四舍五入
round_tensor = torch.round(exp_one)
print('四舍五入round_tensor',round_tensor)

# 计算自然对数
a = torch.randn(5)
b = torch.log(a)
print('以自然数为底的对数',b,b.size())

# 计算除法取余数
# 计算除法余数。 除数与被除数可能同时含有整数和浮点数。此时，余数的正负与被除数相同。
aa =torch.fmod(torch.Tensor([-3, -2, -1, 1, 2, 3]), 2)
print('除法取余数',aa)

# 求平均值
mean_aa = torch.mean(aa,0)
print('平均值',mean_aa)

# 求中位数
median_aa = torch.median(aa,0)
print("中位数", median_aa)

# 求众数
mode_aa = torch.median(a,0)
print("众数",mode_aa)

# 求标准差
std_tensor = torch.std(a,0)
print('标准差',std_tensor)

# 求和
sum_tensor = torch.sum(a)
print("求和",sum_tensor)

#方差
var_a = torch.var(a,0)
print("方差",var_a)

# 两个张量的对应元素值比较，相等为1，不等为0
aa_tensor = torch.eq(a,a)
print("判断相等",aa_tensor)

# 两个张量具有相同的形状和者相同的值，则为True
aa_equal = torch.equal(a,a)
print("相等",aa_equal)

# 找最大值
max_a  = torch.max(a)
print('最大值',max_a)

# 找最小值
min_a = torch.min(a)
print('最小值',min_a)

# torch.Tensor是默认的tensor类型（torch.FlaotTensor)的简称。
aa = torch.FloatTensor([[-1,2,3],[3,-4,5]])
print(aa,type(aa),aa.size())
# 取绝对值覆盖原来的值 func_
aa.abs_()
print(aa,type(aa),aa.size())
# 扩展
x = torch.Tensor([[1],[2],[3]])
print(x.size())
x= x.expand(3,4)
print(x.size(),x)


x = torch.empty(5,3,dtype = torch.long)
print("x",x)

x = torch.Tensor([5.5,3])
print(x)

# reshape
x = torch.randn(4,4)
y = x.view(2,8)
z = x.view(8, -1)
print(x,x.size(),"\n",y,y.size(),"\n",z.size(),z)

# 从numpy到tensor
import numpy as np
a = np.ones(7)
b = torch.from_numpy(a)
print(a,type(a), b,type(b))

# 创建一个跟踪的张量
x = torch.randn(2,2, requires_grad=True)
print(x,type(x))

y = x + 2
print('y',y)
print(y.grad_fn,)

z = y*y*3
out = z.sum()
print(z,type(z),"\n",out,type(out))

a = torch.Tensor([[1,2,3],[1,2,3]])
print('a张量的元素个数',a.numel())
a_list = a.tolist()
print('a_list',a_list,type(a_list),len(a_list))

print('整数等差数列，范围1，6，公差为2',torch.arange(1,6,2))
print('服从标准正态分布生成2行3列的数据:',torch.randn(2,3))

a = torch.randn(3,3,2)
print(a,type(a),a.size())
max_a = torch.max(a)
print('max_a',max_a,type(max_a),max_a.size())

print(torch.max(a,3))

