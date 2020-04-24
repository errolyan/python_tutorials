# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-10-11  23:33
'''

def all_equal(lst):
  return lst[1:] == lst[:-1]

print(all_equal([1, 2, 3, 4, 5, 6]))# False
print(all_equal([1, 1, 1, 1]))# True

def all_unique(lst):
  return len(lst) == len(set(lst))

x = [1,2,3,4,5,6]
y = [1,2,2,3,4,5]
print(all_unique(x) )# True
print(all_unique(y) )# False


# 列表生成式
L = [x*2 for x in range(5)]
print(L)

# 生成器
'''

'''
G = (x*2 for x in range(5))
print(G)
print(next(G),next(G),next(G))

for i in G:
  print(i)

# 迭代器
'''
迭代器是一种访问集合元素的方式。迭代器是一个可以记住遍历的位置的对象。迭代器对象从集合的第一个元素
开始访问，直接到所有的元素被访问完结束。迭代器智能往前不能后退。

'''

# 外函数
def test(number):
    print('number0',number)
    # 内函数
    def test_in(number_in):
        print('number1', number)
        print("in test_in 函数，number_in is %d"%number_in)
        print('number2', number)
        return number+number_in
    return test_in
# 给test函数赋值，这个20就是number
ret = test(20)

# 注意这个100其实给参数nember_in
print(ret(100))

ret = test(50)
# 注意这里的200其实给参数number_in
print(ret(200))


def line_conf(a, b):
  def line(x):
    return a * x + b

  return line


line1 = line_conf(1, 1)
line2 = line_conf(4, 5)
print(line1(5))
print(line2(5))



# 装饰器实例2
from time import ctime,sleep

def timefun(func):
    def wrappedfunc():
        print("%s called at %s"%(func.__name__,ctime()))
        func()
    return wrappedfunc

@timefun
def foo():
    print("I am foo")

foo()
sleep(3)
foo()
