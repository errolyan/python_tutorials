# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： socket_example
   Description :  1、安装saka仓库
                    brew tap rangaofei/saka
                  2、安装软件
                    brew install sokit
                    因为要依赖qt，所以安装会稍微慢一点，安装成功后执行命令即可：
                  3、启动
                    sokit

   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/30  18:44
-------------------------------------------------
   Change Activity:
                  2019/12/30  18:44:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import socket


def main():
    # 1. 创建tcp的套接字
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 2. 链接服务器
    # tcp_socket.connect(("192.168.33.11", 7890))
    server_ip = input("请输入要链接的服务器的ip:")
    server_port = int(input("请输入要链接的服务器的port:"))
    server_addr = (server_ip, server_port)
    tcp_socket.connect(server_addr)

    # 3. 发送数据/接收数据
    send_data = input("请输入要发送的数据:")
    tcp_socket.send(send_data.encode("utf-8"))

    # 4. 关闭套接字
    tcp_socket.close()


if __name__ == "__main__":
    main()