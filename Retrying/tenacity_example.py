# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： tenacity_example
   Description :  重试
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/6  23:29
-------------------------------------------------
   Change Activity:
                  2020/2/6  23:29:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import random
from tenacity import retry

@retry
def do_something_unreliable():
    if random.randint(0, 10) > 1:
        raise IOError("Broken sauce, everything is hosed!!!111one")
    else:
        return "Awesome sauce!"

print(do_something_unreliable())