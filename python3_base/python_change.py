# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol
Email:2681506@gmail.com
Wechat:qq260187357
Date:2019-04-27--14:37
Describe: python base
'''

from pathlib import Path
import pysnooper
@pysnooper.snoop()
def test_fun():
    path_pwd = Path("/Users/yanerrol/Desktop/python3_tutorials/python3_base/")
    print(path_pwd)
    print(path_pwd.exists())
    print(path_pwd.is_dir())
    # path_pwd.chmod(777) #修改目录权限

    # path_pwd.rmdir() # 删除路径

    s = "hello"
    print(len(s))
    print(s[:2])


test_fun()
