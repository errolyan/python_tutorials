# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  
@Email:2681506@gmail.com   
@Date:  2019-05-28  17:17
@File：thread_server.py
@Describe:多线程
'''

import socket
import threading
import time

EOL1 = '\n\n'
EOL2 = '\n\r\n'
body = '''Hello, world! <h1> from the5fire 《Django企业开发实战》</h1> - from {thread_name}'''
response_params = [
    'HTTP/1.0 200 OK',
    'Date: Sat, 10 jun 2017 01:01:01 GMT',
    'Content-Type: text/plain; charset=utf-8',
    'Content-Length: {length}\r\n',
    body,
]
response = b'\r\n'.join(response_params)

def handle_connection(conn, addr):
    request = ""
    while EOL1 not in request and EOL2 not in request:
        request += conn.recv(1024)
    # print(request)
    current_thread = threading.currentThread()
    content_length = len(body.format(thread_name=current_thread.name))
    print(current_thread.name, '-------', 'sleep 10', int(time.time()))
    time.sleep(10)
    conn.send(response.format(thread_name=current_thread.name, length=content_length))
    conn.close()

def main():
    # socket.AF_INET    用于服务器与服务器之间的网络通信
    # socket.SOCK_STREAM    基于TCP的流式socket通信
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置端口可复用，保证我们每次Ctrl C之后，快速再次重启
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind(('127.0.0.1', 8080))
    # 可参考：https://stackoverflow.com/questions/2444459/python-sock-listen
    serversocket.listen(10)
    print('http://127.0.0.1:8080')

    try:
        i = 0
        while True:
            i += 1
            conn, address = serversocket.accept()
            t = threading.Thread(target=handle_connection, args=(conn, address), name='thread-%s' % i)
            t.start()
    finally:
        serversocket.close()

if __name__ == '__main__':
    main()