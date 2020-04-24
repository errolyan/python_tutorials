# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-08-14  14:35
'''

import cv2
import numpy as np

img = cv2.imread("../data/test/test15.bmp")
px = img[0,60]
print(px)

blue = img[40,40,0]
print(blue )

img1=cv2.imread('111.png')
img2=cv2.imread('111.png')
dst=cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindow()