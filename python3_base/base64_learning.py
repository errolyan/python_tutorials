# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-11  00:18
'''

import base64
print( base64.b64encode(b'i\xb7\x1d\xfb\xef\xff'))

import hashlib

md5 = hashlib.md5()
md5.update('how to use md5 in python hashlib?'.encode('utf-8'))
print(md5.hexdigest())

md5.update('how to use md5 in '.encode('utf-8'))
md5.update('python hashlib?'.encode('utf-8'))
print(md5.hexdigest())

print(sum(range(1,101)))

dic1 = {"a":"A","b":"B"}
dic2 = {"c":"C","D":"d"}
valuess = dic1.values()
print(valuess)