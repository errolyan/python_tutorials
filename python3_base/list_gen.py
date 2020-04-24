# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-29--15:23
Describe:列表生成式
'''

import time
lis = [1,2,4,6,77,88,99,100]
res = [i for i in lis if i%2 == 0]
print (res)

lis1 = [22,22,33,45,]
lis.extend(lis1)
lis.sort(reverse=False)
print (lis)

