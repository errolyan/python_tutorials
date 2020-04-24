# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  用opencv 读写显示一张图片
@Evn     :  
@Date    :  2019-08-14  09:10
'''

import cv2
import numpy as np

img_path = "../data/test/11.jpg"
img = cv2.imread(img_path, cv2.WINDOW_NORMAL)
cv2.imshow("test",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("./restore.jpg",img)

from matplotlib import pyplot as plt
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()