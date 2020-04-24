# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-09-16  13:51
'''

import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('./test.jpeg')

#可以检测整个文件夹
# preds = fa.get_landmarks_from_directory('../test/assets/')
preds = fa.get_landmarks(input)

print(preds)

