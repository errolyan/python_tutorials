# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： test_log
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2020/1/10  23:34
-------------------------------------------------
   Change Activity:
                  2020/1/10  23:34:
-------------------------------------------------
'''
__author__ = 'yanerrol'
import logging

logger = logging.getLogger('songs generations server')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(filename="./songs_maker.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("------get request data------")
id =100
logger.info("------get request data id=%d------" % id)

name = "yan"
logger.info("------get request data id={} name={}------".format(id, name))
audio_status, wav_path = 1,"/root.wav"
logger.info("------get request data audiostatus={} wav_path='{}'------".format(audio_status, wav_path))

logger.info("------get request data id={} wav_path='{}' audio_status={}------".format(id, wav_path, audio_status))