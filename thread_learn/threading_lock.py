# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： threading_lock
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/31  14:53
-------------------------------------------------
   Change Activity:
          2020/3/31 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import threading
import os
import time
locka = threading.Lock()
a=0
def add1():
    global a
    try:
        locka.acquire() # 获得锁
        tmp = a + 1
        time.sleep(0.2) # 延时 0.2 秒，模拟写入所需时间
        a = tmp
    finally:
        locka.release() # 释放锁
    print('%s adds a to 1: %d' % (threading.current_thread().getName(), a))

threads = [threading.Thread(name='t%d'%(i,),target=add1) for i in range(10)]
[t.start() for t in threads]