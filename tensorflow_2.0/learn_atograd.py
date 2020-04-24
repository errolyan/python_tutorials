# -*- coding:utf-8 -*-
# /usr/bin/python
'''
-------------------------------------------------
   File Name   ： learn_atograd
   Description :  
   Envs        :  
   Author      :  yanerrol
   Date        ： 2019/12/31  09:02
-------------------------------------------------
   Change Activity:
                  2019/12/31  09:02:
-------------------------------------------------
'''
__author__ = 'yanerrol'
import tensorflow as tf

# 创建4个张量
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)


with tf.GradientTape() as tape:# 构建梯度环境
	tape.watch([w]) # 将w加入梯度跟踪列表
	# 构建计算过程
	y = a * w**2 + b * w + c
# 求导
[dy_dw] = tape.gradient(y, [w])
print('dy_dw',dy_dw,type(dy_dw))


