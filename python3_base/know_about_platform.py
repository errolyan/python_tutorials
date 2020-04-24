# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol  Email:2681506@gmail.com   Wechat:qq260187357
Date:2019-05-10--08:27
File：know about plat
Describe: 了解platform
'''
print (__doc__)



# 获取系统平台
import os
import platform as plat

system_type = plat.system()
print (system_type)

# 判断路径是否存在
modelpath = "./test"
if(not os.path.exists(modelpath)):
    os.makedirs(modelpath)