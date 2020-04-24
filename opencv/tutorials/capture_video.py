# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-08-14  09:25
'''

import cv2
import numpy as np

# 创建相机
def camera_video(time):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    if cap.isOpened() == False:
        cap.open()
    else:
        print('camera is open!')
    while True:
        # capture frame-by-frame
        ret,frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)
            out.write(frame)

            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame',gray)

            if cv2.waitKey(time):# & oxFF ==ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


time = 10000
camera_video(time)