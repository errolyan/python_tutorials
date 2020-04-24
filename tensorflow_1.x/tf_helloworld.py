#! /usr/bin/python3
#__Author__="ErrolYan"
#__Date__="2019.04.06"

import tensorflow as tf

hello = tf.constant("Hello,Tensorflow")

# start tf session

sess = tf.Session()

# run the op

print(sess.run(hello))
