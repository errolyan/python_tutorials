# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  返回函数
@Evn     :  
@Date    :  2019-09-09  16:58
'''

def lazy_sum(*args):
    '''
    返回函数：调用函数但是不需要立刻执行函数
    :param args: 调用的参数
    :return: 真正意义的函数名
    '''
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum

func = lazy_sum(1,3,5,7,9)
print(func())