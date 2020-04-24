# !/usr/bin/python3
# -*-coding:UTF-8-*-

import socket
import ssl


def web_connect():
    sock = ssl.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    sock.connect(('www.sina.com.cn', 443))

    sock.send('GET / HTTP/1.1\r\n'.encode())
    sock.send('Host: www.sina.com.cn\r\n'.encode())
    sock.send(
        'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:68.0) Gecko/20100101 Firefox/68.0\r\n'.encode())
    sock.send('Connection: close\r\n\r\n'.encode())

    buffer = []
    while True:
        d = sock.recv(10240)
        if d:
            buffer.append(d)
        else:
            break
    data = b''.join(buffer)
    # print(data.decode('utf-8'))
    sock.close()

    header, html = data.split(b'\r\n\r\n', 1)  # HTTP头和网页分离
    print(header.decode('utf-8'))
    with open('web_got.html', 'wb') as f:  # 接收的数据写入文件
        f.write(html)


if __name__ == '__main__':
    web_connect()