# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  
@Email:2681506@gmail.com   
@Wechat:qq260187357
@Date:  2019-05-18  23:26
@File：puttext.py
@Describe:加英文水印
'''
print(__doc__)

import cv2
from PIL import Image,ImageFont,ImageDraw
import numpy as np

img = cv2.imread("../data/test/test12.bmp",cv2.IMREAD_COLOR)

text = "helloWorld"
pos = (450,76)
font_type = 4
font_size = 2
color = (255,0,0)
bold = 1

cv2.putText(img,text,pos, font_type, font_size, color,bold)
cv2.imshow('www',img)
cv2.waitKey(0)


'''
加中文水印
'''

img1 = cv2.imread("../data/test/test12.bmp",cv2.IMREAD_COLOR)
pil_image = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
font = ImageFont.truetype('NotoSerifCJK-Regular.ttc', 40)
color = (0,0,255)
pos = (30,64)
text = u"Linux公社www.linuxidc.com"
draw = ImageDraw.Draw(pil_image)
draw.text(pos,text,font=font,fill=color)
cv_img = cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)
cv2.imshow('www',cv_img)
cv2.waitKey(0)
