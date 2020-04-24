# coding=utf-8
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-11--23:32
Describe:
'''
import time

def quick_sort (alist,start,end):
    """快速排序"""
    # 递归的退出条件
    if start >= end:
        return
    mid = alist[start]

    # low 为序列左边的由左向右移动的游标
    low = start

    # high为序列右边的游标
    high = end
    while low < high:
            # 如果low与high未重合，high指向的元素不比基准元素小，则high向左移动
            while low < high and alist[high] >= mid:
                high -= 1
            # 将high指向的元素放到low的位置上
            alist[low] = alist[high]

            # 如果low与high未重合，low指向的元素比基准元素小，则low向右移动
            while low < high and alist[low] < mid:
                low += 1
            # 将low指向的元素放到high的位置上
            alist[high] = alist[low]

    # 退出循环后，low与high重合，此时所指位置为基准元素的正确位置
    # 将基准元素放到该位置
    alist[low] = mid

    # 对基准元素左边的子序列进行快速排序
    quick_sort(alist, start, low-1)

    # 对基准元素右边的子序列进行快速排序
    quick_sort(alist, low+1, end)
    return alist
def main():
    in_list = [99,11,0,22,44,56,1,12,21,434,90]
    start_time =time.time()
    print("start time",start_time)
    in_list = quick_sort(in_list,0,len(in_list)-1)
    print (in_list)
    end_time = time.time()
    print("end time",end_time)


if __name__ == "__main__":
    main()