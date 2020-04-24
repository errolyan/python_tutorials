# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-07-25  15:01
'''

import requests
user_info = {'name': 'foobar', 'password': 'kidding'}
r = requests.post("http://127.1:8080/register", data=user_info)
print (r.text)