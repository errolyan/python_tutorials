# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''

from scipy import misc
from scipy.misc import imsave
f = misc.face(gray = True)
imsave('face.png',f)

import matplotlib.pyplot as plt
#plt.imshow(f)


'''倒置'''
import numpy as np

flip_ud_image = np.flipud(f)
plt.imshow(flip_ud_image)
plt.show()

'''
旋转
'''
from scipy import ndimage
rotate_face = ndimage.rotate(f,45)
plt.imshow(rotate_face)
plt.show()

'''
滤镜是图象增强或者修改的技术
'''
from scipy import misc
face = misc.face()
blurred_face = ndimage.gaussian_filter(face,sigma =3) # sigma值表示5级模糊程度
plt.imshow(blurred_face)
plt.show()

'''
边缘检测
边缘检测是一种用于查找图象内物体边界的图象处理技术。它通过检测亮度不连续性来工作。边缘检测用于诸如图象处理，计算
机视觉和机器视觉领域的图象分割和数据提取。
'''
import scipy.ndimage as nd
import numpy as np
im = np.zeros((256,256))
im[64:-64,64:-64]=1
im[90:-90,90:-90]=2
im = ndimage.gaussian_filter(im,8)

# 检测边缘
sx = ndimage.sobel(im,axis =0,mode ='constant')
sy = ndimage.sobel(im,axis =1, mode = 'constant')
sob = np.hypot(sx,sy)
plt.imshow(sob)
plt.show()