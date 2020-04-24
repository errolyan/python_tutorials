# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： redis_example
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/30  14:58
-------------------------------------------------
   Change Activity:
                  2019/12/30  14:58:
-------------------------------------------------
'''
__author__ = 'yanerrol'
#coding=utf-8
from redis import *

if __name__=="__main__":
    try:
        #创建StrictRedis对象，与redis服务器建立连接
        sr=StrictRedis('10.1.8.83',  '6379')
        #获取键py1的值
        result = sr.get('py1')
        #输出键的值，如果键不存在则返回None
        print (result)
        result = sr.get('py1')
        # 输出键的值，如果键不存在则返回None
        print(result)
        result = sr.set('py1', 'hr')
        # 输出响应结果，如果操作成功则返回True，否则返回False
        print(result)
        # 获取所有的键
        result = sr.keys()
        # 输出响应结果，所有的键构成一个列表，如果没有键则返回空列表
        print(result)
        result = sr.delete('py1')
        # 输出响应结果，如果删除成功则返回受影响的键数，否则则返回0
        print(result)

    except Exception as e:
        print (e)