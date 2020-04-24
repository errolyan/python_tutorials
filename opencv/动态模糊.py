# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： 动态模糊
   Description :  运动模糊：由于相机和物体之间的相对运动造成的模糊，又称为动态模糊
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/30  10:34
-------------------------------------------------
   Change Activity:
                  2019/12/30  10:34:
-------------------------------------------------
'''
__author__ = 'yanerrol'
# coding: utf-8
import numpy as np
import cv2

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

img = cv2.imread('test.png')
img_ = motion_blur(img)

cv2.imshow('Source image',img)
cv2.imshow('blur image',img_)
cv2.waitKey()


'''高斯模糊'''
# coding: utf-8
import numpy as np
import cv2

img = cv2.imread('test.png')
img_ = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0, sigmaY=0)

cv2.imshow('Source image',img)
cv2.imshow('blur image',img_)
cv2.waitKey()