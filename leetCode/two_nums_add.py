# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： two_nums_add
   Description :  AIM:
                  Functions: 1.
                                Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
                                Output: 7 -> 0 -> 8
                                Explanation: 342 + 465 = 807.
   Envs        :  python == 3.5
                  pip install  -i https://pypi.douban.com/simple
   Author      :  yanerrol
   Date        ： 2020/3/1  00:59
-------------------------------------------------
   Change Activity:
          2020/3/1 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

class Solution():
    def addTwoNumbers(self,list_a,list_b):
        '''Add'''
        list_a, list_b = list(list_a), list(list_b)
        list_c = []
        vlues_a = 0
        if len(list_a) >= 1:
            for i in list_a:
                vlues_a += i * 10 ** list_a.index(i)

        vlues_b = 0
        if len(list_b) >= 1:
            for i in list_b:
                vlues_b += i * 10 ** list_b.index(i)

        vlues_c = vlues_a + vlues_b
        while vlues_c >= 10 :
            list_c.append(vlues_c % 10)
            vlues_c = vlues_c // 10
        list_c.append(vlues_c)

        return list_c

if __name__ == '__main__':
    list_a = [2,4,3]
    list_b = [5,6,4]
    new = Solution()
    print(new.addTwoNumbers(list_a, list_b))