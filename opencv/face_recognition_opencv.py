# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :  2019-10-12  09:06
'''

import cv2
import matplotlib.pyplot as plt
import cvlib as cv

img_path = "data/test/12.jpg"

im = cv2.imread(img_path)
faces, confidences = cv.detect_face(im)
for face in faces:
    (startX, startY) = face[0],face[1]
    (endX, endY) = face[2],face[3]
    cv2.rectangle(im, (startX, startY), (endX, endY), (0,255,0),2)
plt.imshow(im)
plt.show()

