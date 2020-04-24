# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol  Email:2681506@gmail.com   Wechat:qq260187357
Date:2019-05-08--13:48
File：img2binary.py
Describe:读取图片，图片灰度化处理，灰度处理后的图片二值化
'''

print (__doc__)

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def read_img(path):
    # Load an color image in grayscale
    img = cv.imread(path)
    img = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return img

def imgs2gray(img):
    im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return im_gray

def imgs2binary(img):
    im_at_mean = cv.adaptiveThreshold(img,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5,
                                       10)  # 使用自适应阈值进行二值化处理，其他二值化方法可查询API使用
    return im_at_mean

def imgs2invert(img):
    #invert the image
    print("img",img,type(img))
    im_revert = 255 - img

    return im_revert

def imgs2denoising(img):
    im_denois = cv.GaussianBlur(img, (9, 9),0) # 15, 8, 25
    return im_denois

def show_img(img,name):
    # matplot plt show img
    print(name,img, type(img))
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(name)
    plt.show()

def img2segmentation(gray):
    gradX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0)
    gradY = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1)

    gradient = cv.subtract(gradX, gradY)
    gradient = cv.convertScaleAbs(gradient)
    return gradient

def gradient2denoise(gradient):
    blurred = cv.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)
    print("thresh",thresh,type(thresh))
    return thresh

def img_box(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    closed = cv.erode(closed, None, iterations=4)
    closed = cv.dilate(closed, None, iterations=4)
    (_, cnts, _) = cv.findContours(
    # 参数一： 二值化图像
    closed.copy(),
    # 参数二：轮廓类型
    cv.RETR_EXTERNAL,             #表示只检测外轮廓
    # cv2.RETR_CCOMP,                #建立两个等级的轮廓,上一层是边界
    # cv2.RETR_LIST,                 #检测的轮廓不建立等级关系
    # cv2.RETR_TREE,                 #建立一个等级树结构的轮廓
    # cv2.CHAIN_APPROX_NONE,         #存储所有的轮廓点，相邻的两个点的像素位置差不超过1
    #参数三：处理近似方法
    cv.CHAIN_APPROX_SIMPLE,         #例如一个矩形轮廓只需4个点来保存轮廓信息
    # cv2.CHAIN_APPROX_TC89_L1,
    # cv2.CHAIN_APPROX_TC89_KCOS
    )
    c = sorted(cnts, key=cv.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv.minAreaRect(c)
    box = np.int0(cv.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    draw_img = cv.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
    show_img(draw_img, "draw_img")


def main():
    path = "../data/test/test15.bmp"
    img = read_img(path)
    name = "original_img"
    show_img(img,name)

    im_gray = imgs2gray(img)
    name = "gray_img"
    show_img(im_gray,name)

    im_at_mean = imgs2binary(im_gray)
    name = "binary_img"
    show_img(im_at_mean,name)

    im_revert = imgs2invert(im_at_mean)
    name = "invert_img"
    show_img(im_revert,name)

    im_denois = imgs2denoising(im_revert)
    name = "im_denois"
    show_img(im_denois,name)

    im_gradient = img2segmentation(im_denois)
    name = "im_gradient"
    show_img(im_gradient,name)

    thresh = gradient2denoise(im_gray)
    name  = "thresh"
    show_img(thresh,name)

    # im_thresh_mean = imgs2binary(thresh)
    # name = "im_thresh_mean"
    # show_img(im_thresh_mean, name)

    # img_box(im_thresh_mean)



if __name__ == "__main__":
    main()