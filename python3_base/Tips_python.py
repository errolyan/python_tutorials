# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： Tips_python
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 3。5
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/30  14:24
-------------------------------------------------
   Change Activity:
          2020/3/30 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

# 合并字典
print({**{'a':1,'b':2}, **{'c':3}})

# 求绝对值
print(abs(-6))

# all 用法
print (all([1,2,3]))
print(all([0,12]))

# repr 展示对象
class People(object):
    def __init__(self,id,name):
        self.id = id
        self.name = name

    def __repr__(self):
        return 'id = '+self.id + ', name = '+self.name

    def __call__(self):
        print('I can be called')

zhangsan = People(id='007',name='zhangsan')
print(zhangsan)

# 判断对象是否含有这个属性
print('shuxing',hasattr(zhangsan,"shuxing"))

# 自定义的对象都是可以哈希的
print('对象都是可以哈希',hash(zhangsan))

# 获取对象的id
print("zhangsan de ID",id(zhangsan))
# list  dict set 是不可以hash的

# 十进制转二进制
print(bin(10))

# 十进制转八进制
print(oct(10))

# 十进制转十六进制
print(hex(15))

# 字符串转字节
string = "New world"
print(bytes(string,encoding="utf-8"))
print(bytes(string,encoding="gbk"))

# 数字转字符
i = 100.001
str(i)

# 转化为浮点数
print(float(3))

# 判断是否可以调用
print(callable(string))
print(callable(str))
print('zhangsan',zhangsan())
print(callable(int))

print('---------------------tip 1--------------------')

# 复数
print(complex(1,2))

# 字典压缩
zip_test = zip(('a','b','c'),[1,2,3])
print(zip_test,type(zip_test))
dict1 = dict(zip(('a','b','c'),[1,2,3]))
print(dict1)

# 查看类的所有方法
print(dir(zhangsan))

# 取商和取余
print(divmod(10,3))

# 计算
s = " 1 + 2 * 3 / 2"
print(eval(s))

# 查看变量占用字节
import sys
a = {'a':1}
print(sys.getsizeof(a))

# 过滤器
fil_data = filter(lambda x:x>10,[1,11,2,45,7,6,13])
print(fil_data,type(fil_data))
print(list(fil_data))

# 字符串格式化
print("id:{1},name:{0}".format('LiSi',"0001"))

# 冻结集合
lis1 = [1,1,2,23,44]
print(frozenset(lis1)),lis1.pop(1)

# 获取输入
# input_data = input("请输入？")
# print(input_data)

# object是所有对象的基类
o = object()
print(type(o))