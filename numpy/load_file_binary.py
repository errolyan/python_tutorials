# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： load_file_binary
   Description :  AIM: 
                  Functions: 1. 
                             2. 
   Envs        :  python == 
                  pip install  -i https://pypi.douban.com/simple 
   Author      :  yanerrol
   Date        ： 2020/3/27  10:11
-------------------------------------------------
   Change Activity:
          2020/3/27 : text
-------------------------------------------------
'''
__author__ = 'yanerrol'

import numpy

def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = numpy.fromfile(fid_lab, dtype=numpy.float32)
    fid_lab.close()
    print('features',features,type(features))
    print('features.size', features.size, '\n', float(dimension), '\n', 'file_name', file_name, '\n')
    if features.size % int(dimension) != 0:
        features = features[0:features.size - features.size % int(dimension)]
    assert features.size % float(dimension) == 0.0, 'specified dimension %s not compatible with data' % (dimension)
    frame_number = features.size // dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return features, frame_number


if __name__ == '__main__':
    dimension = 5
    file_name = 'nitech_jp_song070_f001_010.bap'
    load_binary_file_frame(file_name, dimension)