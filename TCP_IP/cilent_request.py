# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-11  23:51
'''
# !/usr/bin/python3
# -*-coding:UTF-8-*-

import socket

def clint():
    '''客户端程序'''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 9999))    # 建立连接
    print(s.recv(1024).decode('utf-8'))    # 接收消息
    for data in [b'Michael', b'Tracy', b'Sarah']:
        s.send(data)    # 发送数据
        print(s.recv(1024).decode('utf-8'))
    s.send(b'exit')
    s.close()

if __name__ == '__main__':
    clint()


