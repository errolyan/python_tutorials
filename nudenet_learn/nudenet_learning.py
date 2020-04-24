# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： nudenet_learning
   Description :  检黄大师
   Envs        :  py37
   Author      :  yanerrol
   Date        ： 2019/12/15  23:25
-------------------------------------------------
   Change Activity:
                  2019/12/15  23:25:
-------------------------------------------------
'''
__author__ = 'yanerrol'

from nudenet import NudeDetector
from nudenet import NudeClassifier
classifier = NudeClassifier('classifier_checkpoint_path')
classifier.classify('path_to_nude_image')
# {'path_to_nude_image': {'safe': 5.8822202e-08, 'nude': 1.0}}

detector = NudeDetector('detector_checkpoint_path')

# Performing detection
detector.detect('path_to_nude_image')
# [{'box': [352, 688, 550, 858], 'score': 0.9603578, 'label': 'BELLY'}, {'box': [507, 896, 586, 1055], 'score': 0.94103414, 'label': 'F_GENITALIA'}, {'box': [221, 467, 552, 650], 'score': 0.8011624, 'label': 'F_BREAST'}, {'box': [359, 464, 543, 626], 'score': 0.6324697, 'label': 'F_BREAST'}]

# Censoring an image
detector.censor(
    'path_to_nude_image',
    out_path='censored_image_path',
    visualize=False)
