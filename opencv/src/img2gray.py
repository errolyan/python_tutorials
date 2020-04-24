# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol  Email:2681506@gmail.com   Wechat:qq260187357
Date:2019-05-08--10:07
File：img2gray.py
Describe:将加载的图象进行灰度化处理
'''

print (__doc__)

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def read_img(path):
    # Load an color image in grayscale
    img = cv.imread(path, 0)
    return img

def show_img(img):
    # matplot plt show img
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def show_Digit(vector):
    # 重新变形
    img = vector.reshape((8, 8))
    plt.imshow(img, cmap='gray')
    plt.show()



def img2vect(img):
    # 将灰度图变为向量
    # 变换为 8×8
    img = cv.resize(img, (8, 8), interpolation=cv.INTER_LINEAR)
    # 变为向量 并将数值放缩在 0-16之间
    return np.reshape(img, (64)) / 16

def write_img(img):
    # save img
    cv.imwrite('../data/test/test13.bmp', img)

def main():
    path = "../data/test/test15.bmp"
    img = read_img(path)
    vector = img2vect(img)
    print (vector)
    show_img(img)
    write_img(img)


if __name__=="__main__":
    main()
