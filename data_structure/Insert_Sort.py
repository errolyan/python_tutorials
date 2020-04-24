# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-11--23:32
Describe:Insert Sort
'''
import time

def insert_sort(alist):
    # 从第二个位置，即下标为1的元素开始向前插入
    for i in range(1, len(alist)):
        # 从第i个元素开始向前比较，如果小于前一个元素，交换位置
        for j in range(i, 0, -1):
            if alist[j] < alist[j-1]:
                alist[j], alist[j-1] = alist[j-1], alist[j]
    return alist

def main():
    in_list = [99.11,0,22,44,56,1,12,21,434,90]
    start_time =time.time()
    print("start time",start_time)
    in_list = insert_sort(in_list)
    print (in_list)
    end_time = time.time()
    print("end time",end_time)

if __name__ == "__main__":
    main()