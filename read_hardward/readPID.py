# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  读取进程
@Evn     :
@Date    :  2019-07-02  11:20
'''


import os
import time
import psutil
import sys
import atexit

# the process name
PROCESS_NAME1 = "dwm.exe"
PROCESS_NAME2 = "QQ.exe"
PROCESS_NAME3 = "WeChat.exe"
PROCESS_NAME4 = "wininit.exe"
PROCESS_NAME5 = "360Tray.exe"
PROCESS_NAME6 = "conhost.exe"


# function of cpu total state
def GetCPUstate(time_count=1):  # cpu物理个数   +   cpu总使用率
    return (str(psutil.cpu_count(logical=False)) + "-" + str(psutil.cpu_percent(time_count, 0)) + "%")


# function of evryone state
def GetCPUsatus(time_count=1):  # 每个cpu的使用率
    return (str(psutil.cpu_percent(time_count, 1)) + "%")


# function of memory state
def GetMemorystate():
    phymem = psutil.virtual_memory()
    string = str(int(phymem.total / 1024 / 1024)) + "M"  # 内存总数
    string += "-"
    string += str(int(phymem.used / 1024 / 1024)) + "M"  # 已使用内存
    string += "-"
    string += str(int(phymem.free / 1024 / 1024)) + "M"  # 剩余内存
    string += "-"
    sum_mem = str(int(phymem.used / 1024 / 1024) / int(phymem.total / 1024 / 1024) * 100)  # 内存使用率
    string += sum_mem[0:5] + "%"
    return (string)


# function of disk state
def GetDisksatate():
    diskinfo = psutil.disk_usage('/')
    disk_str = str(int(diskinfo.total / 1024 / 1024 / 1024)) + "G"  # 硬盘总容量
    disk_str += "-"
    disk_str += str(int(diskinfo.used / 1024 / 1024 / 1024)) + "G"  # 已使用硬盘容量
    disk_str += "-"
    disk_str += str(int(diskinfo.free / 1024 / 1024 / 1024)) + "G"  # 剩余容量
    disk_str += "-"
    sum_disk = str(int(diskinfo.used / 1024 / 1024 / 1024) / int(diskinfo.total / 1024 / 1024 / 1024) * 100)  # 硬盘使用率
    disk_str += sum_disk[0:5] + "%"
    return (disk_str)


# main
def GetInfoMain():
    time_count = 1
    infomaition = GetCPUstate(time_count)
    infomaition += "-"
    infomaition += GetCPUsatus(time_count)
    infomaition += "-"
    infomaition += GetMemorystate()
    infomaition += "-"
    infomaition += GetDisksatate()
    infomaition += "-"


    print (infomaition)
    return (infomaition)

GetInfoMain()
try:
   while 1:
       time.sleep(1)
       strstr=GetInfoMain()
       print (strstr)
except:
  print ("exit bd-CState.py")