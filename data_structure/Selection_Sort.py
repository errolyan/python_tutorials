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

def Selection_Sort(in_list):
    n = len(in_list)
    # 需要进行n-1次排序
    for i in range(n-1):
        # 记录最下的位置
        min_index = i
        # 从i+1位置到末尾选择出最小的数据
        for j in range(i+1,n):
            if in_list[j] <in_list[min_index]:
                min_index = j
        # 如果选择出的数据不再正确的位置，进行交换
        if min_index != i:
            in_list[i],in_list[min_index] = in_list[min_index],in_list[i]
    return in_list
def main():
    in_list = [99.11,0,22,44,56,1,12,21,434,90]
    start_time =time.time()
    print("start time",start_time)
    in_list = Selection_Sort(in_list)
    print (in_list)
    end_time = time.time()
    print("end time",end_time)

if __name__ == "__main__":
    main()