# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-11--23:32
Describe: Bubble Sort:简单的来说就是将最小的数字排到最前面，大的数字排后面。
'''
import time

def Bubble_sort(in_list):
    leng = len(in_list)
    for j in range(leng-1,0,-1):
        for i in range(j):
            if in_list[i]>in_list[i+1]:
                in_list[i],in_list[i+1] = in_list[i+1],in_list[i]

    return in_list

def main():
    input_list = [11,33,66,1,55,88,99,0]
    start_time = time.time()
    input_list = Bubble_sort(input_list)
    print(input_list)
    end_time = time.time()
    print(end_time - start_time)

if __name__ == "__main__":
    main()