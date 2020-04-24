# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： client_tcp
   Description :
   Envs        :
   Author      :  yanerrol
   Date        ： 2019/12/15  15:35
-------------------------------------------------
   Change Activity:
                  2019/12/15  15:35:
-------------------------------------------------
'''
__author__ = 'yanerrol'
import socket
if __name__ == '__main__':
    # 创建套接字
    tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 和服务端连接
    server_ip = input("输入服务器IP：")
    tcp_client_socket.connect((server_ip, 3356))
    # 发送下载文件的请求
    while True:
        file_name = input("请输入要下载的文件名：")
        # 编码
        file_name_data = file_name.encode("utf-8")
        # 发送文件下载请求数据
        tcp_client_socket.send(file_name_data)
        #　接收要下载的文件信息
        file_info = tcp_client_socket.recv(1024)
        # 文件信息解码
        info_decode = file_info.decode("utf-8")
        print(info_decode)
        # 获取文件大小
        fileszie = float(info_decode.split('：')[2].split('MB')[0])
        fileszie2 = fileszie * 1024
        # 是否下载？输入ｙ　确认　输入ｑ 取消
        opts = input("是否下载？(y 确认　q 取消)")
        if opts == 'q':
            print("下载取消！程序退出")
        else:
            print("正在下载　》》》")
        # 向服务器确认正在下载
        tcp_client_socket.send(b'y')

        # 把数据写入到文件里
        with open("./" + file_name, "wb") as file:
            # 目前接收到的数据包数目
            cnum = 0
            while True:
                # 循环接收文件数据
                file_data = tcp_client_socket.recv(1024)
                # 接收到数据
                if file_data:
                    # 写入数据
                    file.write(file_data)
                    cnum = cnum + 1
                    jindu = cnum / fileszie2 * 100
                    print("当前已下载：%.2f%%" % jindu, end="\r")
                # 接收完成
                else:
                    print("下载结束！")
                    break
            # 关闭套接字
    tcp_client_socket.close()
