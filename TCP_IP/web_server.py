# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-11  23:49
'''
import socket
import threading
import time

def sever():
    '''服务器程序'''
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 9999))    # 监听端口
    s.listen(5)
    print('Waiting for connection...')

    while True:
        sock, addr = s.accept()    # 接受一个新连接
        t = threading.Thread(target=tcplink, args=(sock, addr))    # 创建新线程来处理TCP连接
        t.start()

def tcplink(sock, addr):
    '''收发程序'''
    print('Accept new connection from %s:%s...' % addr)
    sock.send(b'Welcome!')
    while True:
        data = sock.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8') == 'exit':
            break
        sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    sock.close()
    print('Connection from %s:%s closed.' % addr)

if __name__ == '__main__':
    sever()


