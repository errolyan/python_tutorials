# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： Reverse_Integer
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/1  02:32
-------------------------------------------------
   Change Activity:
          2020/3/1 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

class Solution:
    def int2list(self,x):
        '''int 2 list'''
        list_x = []
        while x >= 10:
            list_x.insert(0,x % 10)
            x = x // 10
        list_x.insert(0,x)
        print(list_x)
        return list_x

    def list2int(self,list_x):
        y = 0
        if len(list_x) >= 1:
            for i in range(len(list_x)):
                y += list_x[i] * 10 ** i
        return y

    def reverse(self, x):
        singed_int = 1
        if x >= 0:
            list_x = self.int2list(x)
        else:
            singed_int = -1
            list_x = self.int2list(x * singed_int)

        y = int(self.list2int(list_x) * singed_int)
        return y

if __name__ == '__main__':
    x =153499999999999999
    new = Solution()
    print(new.reverse(x))


