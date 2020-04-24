# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  @Email:2681506@gmail.com   
@Date:  2019-06-03  10:27
@File：
@Describe:
@Evn:
'''

from flask import Flask,url_for
application = Flask(__name__)

@application.route('/')
def hello_world():
    return 'Hello, World!'

with application.test_request_context():
    print (url_for('static', filename='./output/test.png'))

