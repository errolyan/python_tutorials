# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： pickle_learn
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/23  17:01
-------------------------------------------------
   Change Activity:
          2020/3/23 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import pickle

with open('test.pkl','rb') as file:
    pickled_data = pickle.load(file)


import h5py
filename = 'hhh.hdf5'
data = h5py.File(filename,'r')

import scipy.io
filename = 'work.mat'
mat = scipy.io.loadmat(filename)

import os
path = './Desktop'
wd = os.getwd()
os.listdir(path)
os.chdir(path)
os.rename('origin_name.txt','newname.txt')
os.remove('filename.txt')
os.mkdir('newdir') # 新建文件夹

