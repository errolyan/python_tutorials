#!/usr/bin python
# -*-coding-*-:"utf-8"
#__Author__="Yan Errol"
#__Time__="2019.04.08"
#__Descrbe__="learning tf"

import tensorflow as tf
import numpy as np

ones = tf.ones([2,3],tf.int32)
zeros = tf.zeros([3,3],tf.float32)
print(ones,type(ones))
print(zeros,type(zeros))
print(tf.shape(ones))
print(tf.shape(zeros))
print(tf.size(ones))
print(tf.rank(ones))

x = tf.constant(2)
y = tf.constant(5)
def f1():return tf.multiply(x,17)
def f2():return tf.add(y,23)

r = tf.cond(tf.less(x,y),f1,f2)

print(r)

with tf.Session() as sess:
    print(sess.run(r))

var1 = tf.Variable(1.0,name= "firstvar")
print("var1:",var1.name)
var1 = tf.Variable(2.0,name="firstvar")
print("var1:",var1.name)
var2 = tf.Variable(3.0)
print("var2:",var2.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1 = ",var1.eval())
    print("var2 = ",var2.eval())


# 图的基本操作

c = tf.constant(1.0)

g = tf.Graph()
with g.as_default():
    c1 = tf.constant(0.0)
    print(c1.graph)
    print(g)
    print(c.graph)

    g2 = tf.get_default_graph()
    print(g2)

   # tf.reset_default_graph()
    g3 = tf.get_default_graph()
    print(g3)

    print(c1.name)
    t = g.get_tensor_by_name(name="Const:0")
    print(t)