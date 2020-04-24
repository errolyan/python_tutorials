# -*- coding:utf-8 -*-
# /usr/bin/python
'''
Author:Yan Errol  Email:2681506@gmail.com   Wechat:qq260187357
Date:2019-05-13--09:50
File：opencv_learning.py
Describe: 学习opencv
'''
print(__doc__)

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print(cv2.__version__)

img = cv2.imread("../data/test/test15.bmp")

plt.imshow(img)
#plt.show()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
#plt.show()

# 查看通道是如何构成的
# Plot the three channels of the image
fig, axs = plt.subplots(nrows = 1, ncols = 3,figsize = (20, 20))

for i in range(0, 3):
    ax =axs[i]
    ax.imshow(img_rgb[:, :, i], cmap = 'gray')

plt.show()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray, cmap = 'gray')
plt.show()