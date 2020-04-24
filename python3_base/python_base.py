# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-10-03  12:09
'''

# 将列表中的所有元素组合成字符串
list1 = ["python","is","good"]
print(" ".join(list1))
print(reversed(list1))
for item in reversed(list1):
    print(item)

a = "hello_world"
print(id(a))
print(id("hello"+"_"+"world"))

b = "hello_world"
c = a is b
print(c)
'''
由于 Cpython 在编译优化时, 某些情况下会尝试使用已经存在的不可
变对象而不是每次都创建一个新对象. (这种行为被称作字符串的驻留[string interning])
所有长度为 0 和长度为 1 的字符串都被驻留.

字符串在编译时被实现 ('wtf' 将被驻留, 但是 ''.join(['w', 't', 'f']) 将不会被驻留)

字符串中只包含字母，数字或下划线时将会驻留. 所以 'wtf!' 由于包含 ! 而未被驻留. 可以在这里找到 CPython 对此规则的实现.
'''

a = "hello_world!"
b = "hello_world!"
c = a is b
print(c)
print('a' * 20 is 'aaaaaaaaaaaaaaaaaaaa')

print('a' * 21 is 'aaaaaaaaaaaaaaaaaaaaa')