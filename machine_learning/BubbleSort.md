[TOC]
# 1.bubble sort
1. 冒泡排序（Bubble Sort）是一种典型的交换排序算法，通过交换数据元素的位置进行排序。
## 1.1基本原理
1. **基本思想**
*冒泡排序*的基本思想就是：从无序序列头部开始，进行两两比较，根据大小交换位置，直到最后将最大（小）的数据元素交换到了无序队列的队尾，从而成为有序序列的一部分；下一次继续这个过程，直到所有数据元素都排好序。
*算法的核心* 在于每次通过两两比较交换位置，选出剩余无序序列里最大（小）的数据元素放到队尾
2. 运行过程
```
1、比较相邻的元素。如果第一个比第二个大（小），就交换他们两个。
2、对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大（小）的数。
3、针对所有的元素重复以上的步骤，除了最后已经选出的元素（有序）。
4、持续每次对越来越少的元素（无序元素）重复上面的步骤，直到没有任何一对数字需要比较，则序列最终有序。
```

![算法实例](https://img-blog.csdn.net/20160316103848750)

## python3 版本
```
#! /usr/bin/ python
# coding="utf-8"
# __Author__="ErrolYan"

lis = [1,255,11,334,45,-1,-88]
def bubble_sort(lists):
    for i in range(len(lists)-1):
        for j in range(len(lists)-1-i):
            if lists[j] > lists[j+1]:
                lists[j],lists[j+1]=lists[j+1],lists[j]
    return lists

print(bubble_sort(lis))
```
# 2. n次方的计算
## 2.1 python3实现
```
def power(x,n):
    if x == 0:
        print("计算是非法的")
    result = 1
    while n > 0:
        result *=x
        n -= 1
    return result
print(power(2,4))
```
# 3.n的阶乘的计算
```
def fac():
    num = int(input("请输入一个数："))
    if num < 0:
        print("sorry,负数没有阶乘")
    elif num ==0:
        print("0的阶乘是1！")
    else:
        result = 1
        for i in range(1,num + 1):
            result *= i
        print("%d 的阶乘为 %d" %(num,result))
fac()
```
# 4.列表生成式
```
import os
dir = [f for f in os.listdir("/")]

L = ["HELLO","NI hao","ZgHH"]
l = [s.lower() for s in L]
print(l)
```
# 5.文件夹下的所有文件
```
import os
for i in os.listdir("/"):
    print(os.path.join("/",i))
```
## 4.1查找以“.py"结尾的文件
```
import os
def show_file(path):
    for i in os.listdir(path):
        path_dir = os.path.join(path,i)
        if path_dir.endswith(".py"):
            print(path_dir)
        if os.path.isdir(path_dir):
            show_file(path_dir)
show_file("/media/hacker/ErrolYan/07coding")
```
# 5.字典的键值颠倒
```
dic = {"A":"a","B":"b","C":"c"}
new_dic = {y:x for x,y in dic.items()}
print(new_dic)
```
# 6.替换字典中的固定字符为另外一个字符
```
num = ["numbers","hello",3,35,56,7,9,11]
for i in range(num.count(3)):
    thr_index = num.index(3)
    num[thr_index]="3a"
    print(num)
list_num = l + num
print(list_num)
print(list(set(list_num)))
```
# 7.制作验证码
```
num = ["numbers","hello",3,35,56,7,9,11]
for i in range(num.count(3)):
    thr_index = num.index(3)
    num[thr_index]="3a"
    print(num)
list_num = l + num
print(list_num)
print(list(set(list_num)))
```
# 8.测试二进制八进制，十六进制
```
n = 19
print(bin(n),oct(n),hex(n))
```
# 9.测试判断字符串
```
print("测试判断字符串")
string = "runoob.com"
print(str.isalnum()) #判断所有字符都是数字或者字母
print(str.isalpha()) #判断所有字符都是字母
print(str.isdigit()) #判断所有字符都是数字
print(str.islower()) #判断所有字符都是小写字母
print(str.isupper()) #判断所有字符都是大写字母
print(str.istitle()) #判断所有字符都是首字母大写
print(str.isspace()) #判断所有字符都是空白字符，\t \n \r
```