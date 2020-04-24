# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： socket_server
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/12  13:47
-------------------------------------------------
   Change Activity:
                  2019/12/12  13:47:
-------------------------------------------------
'''
__author__ = 'yanerrol'

import socket
import os
import threading


# 处理客户端请求下载文件的操作（从主线程提出来的代码）
def deal_client_request(ip_port, service_client_socket):
    # 连接成功后，输出“客户端连接成功”和客户端的ip和端口
     print("客户端连接成功", ip_port)
    # 接收客户端的请求信息
     file_name = service_client_socket.recv(1024)
    # 解码
     file_name_data = file_name.decode("utf-8")
    # 判断文件是否存在
    if os.path.exists(file_name_data):
        #输出文件字节数
         fsize = os.path.getsize(file_name_data)
        #转化为兆单位
         fmb = fsize/float(1024*1024)
        #要传输的文件信息
         senddata = "文件名：%s 文件大小：%.2fMB"%(file_name_data,fmb)
        #发送和打印文件信息
         service_client_socket.send(senddata.encode("utf-8"))
        print("请求文件名：%s 文件大小：%.2f MB"%(file_name_data,fmb))
        #接受客户是否需要下载
options = service_client_socket.recv(1024)
if options.decode("utf-8") == "y":
# 打开文件
with open(file_name_data, "rb") as f:
#　计算总数据包数目
 nums = fsize/1024
#　当前传输的数据包数目
 cnum = 0

whileTrue:
 file_data = f.read(1024)
 cnum = cnum + 1
 jindu = cnum/nums*100

 print("当前已下载：%.2f%%"%jindu,end = "\r")