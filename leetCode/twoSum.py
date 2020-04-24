# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： twoSum
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/1  00:48
-------------------------------------------------
   Change Activity:
          2020/3/1 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

class Solution:
    def twoSum(self, nums, target):
        lists = []
        for j in range(len(nums)):
            for i in range(j+1,len(nums)):
                if nums[j]+nums[i] == target:
                    lists.append(j)
                    lists.append(i)
        return lists

if __name__=="__main__":
    new = Solution()
    nums = [3,2,4]
    target = 6
    print(new.twoSum(nums,target))


