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


def power(x,n):
    if x == 0:
        print("计算是非法的")
    result = 1
    while n > 0:
        result *=x
        n -= 1
    return result
print(power(2,4))


# 计算阶乘

def fac():
    #num = int(input("请输入一个数："))
    num = 10
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

import os
dir = [f for f in os.listdir("/")]

L = ["HELLO","NI hao","ZgHH","11"]
l = [s.lower() for s in L]
print(l)

# import os
# def show_file(path):
#     for i in os.listdir(path):
#         path_dir = os.path.join(path,i)
#         if path_dir.endswith(".py"):
#             print(path_dir)
#         if os.path.isdir(path_dir):
#             show_file(path_dir)
# show_file("/media/hacker/ErrolYan/07coding")

dic = {"A":"a","B":"b","C":"c"}
new_dic = {y:x for x,y in dic.items()}
print(new_dic)


num = ["numbers","hello",3,35,56,7,9,11]
for i in range(num.count(3)):
    thr_index = num.index(3)
    num[thr_index]="3a"
    print(num)
list_num = l + num
print(list_num)
print(list(set(list_num)))



import random
def Verification_code():
    list_ma = []
    for i in range(11,99):
        list_ma.append(str(i))
    for j in range(10,99):
        list_ma.append(str(j))
    for k in range(12,98):
        list_ma.append(str(k))
    ma = random.sample(list_ma,3)
    print(ma)
    ma = ''.join(ma)
    print(ma)
Verification_code()

n = 19
print(bin(n),oct(n),hex(n))


print("测试判断字符串")
str = " "
print(str.isalnum()) #判断所有字符都是数字或者字母
print(str.isalpha()) #判断所有字符都是字母
print(str.isdigit()) #判断所有字符都是数字
print(str.islower()) #判断所有字符都是小写字母
print(str.isupper()) #判断所有字符都是大写字母
print(str.istitle()) #判断所有字符都是首字母大写
print(str.isspace()) #判断所有字符都是空白字符，\t \n \r