# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  pickle
@Evn     :  二进制文件读写
@Date    :  2019-09-15  16:32
'''

import pickle
xx = ['aa',4,'[1,2,3]']
yy = pickle.dumps(xx)
with open("test.dat",'wb') as f:
    f.write(yy)

with open('test.dat','rb') as f:
    yy = f.read()

zz = pickle.loads(yy)
print(zz)


