# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： text_re
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/2/18  16:45
-------------------------------------------------
   Change Activity:
          2020/2/18 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import re
sentense = "你好,2020年CSDN！google 世界，时间真的很美好？我是不信了AI。hello,my name is xiaoming ,how are you？"
eng_word_list = re.findall('[a-zA-Z]+', sentense)
print(eng_word_list)

