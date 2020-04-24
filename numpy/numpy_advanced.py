# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： numpy_advanced
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/30  13:39
-------------------------------------------------
   Change Activity:
                  2019/12/30  13:39:
-------------------------------------------------
'''
__author__ = 'yanerrol'


import numpy as np
np.random.seed(12345)
np.set_printoptions(precision=4, suppress=True)
import numpy as np
data = {i : np.random.randn() for i in range(7)}
print(data)