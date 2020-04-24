# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-15  16:48
'''
import uuid

node = uuid.uuid1()
print(type(node),node)
hex = node.hex
print(hex)
mac = hex[-12:]
print(mac)



