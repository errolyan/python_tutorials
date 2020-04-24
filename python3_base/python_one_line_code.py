# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
'''
数学口诀
'''
print('\n'.join([' '.join(['%2s x%2s = %2s'%(j,i,i*j) for j in range(1,i+1)]) for i in range(1,10)]))
#
# '''
# 迷宫
# '''
# print(''.join(__import__('random').choice('\u2571\u2572') for i in range(50*24)))