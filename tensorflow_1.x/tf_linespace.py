# -*- coding:utf-8 -*-
#!/usr/bin/python3
#__Author__="Yan Errol"
# __Time__="2019.04.08"
#__Descrbe__="learning tf"

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

##########################################################################
## 建立数据集                                                                           ##
##########################################################################
# 创建模拟数据集
train_x = np.linspace(-1, 1, 100)
print(train_x)
train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3
#print(*train_x.shape, train_x.shape)
print(np.random.randn(4))
print(train_y)

plt.plot(train_x, train_y, 'ro', label="Original data")
plt.legend()
plt.show()

##########################################################################
## 建立模型                                                                           ##
##########################################################################
# 建立数据流管道 也是占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向网络
z = tf.multiply(X, W) + b

# 反向网络

loss = tf.reduce_mean(tf.square(Y - z))
# 设定学习率
learning_rate = 0.01

# 建立优化器

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

plotdata = {"batchsize": [], "loss": []}


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(
        a[(idx - w):idx]) / w for idx, val in enumerate(a)]

##########################################################################
## 训练模型                                                                           ##
##########################################################################


# 初始化参数
init = tf.global_variables_initializer()

# 定义参数
training_epoch = 200
display_step = 2


# 定义保存权重器
saver = tf.train.Saver()
# 起动session

with tf.Session() as sess:
    sess.run(init)
    plotdata = {"batchsize": [], "loss": []}  # 存放批次值和loss值

    # 创建summary_write,用于写文件
    merged_summary_op = tf.summary.merge_all()  # 合并所有summary
    summary_writer = tf.summary.FileWriter(
        "log/mnist_with_summaries", sess.graph)

    # 向模型输入数据
    for epoch in range(training_epoch):
        for(x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if epoch % display_step == 0:
            loss1 = sess.run(loss, feed_dict={X: train_x, Y: train_y})
            print(
                "Epoch:",
                epoch + 1,
                "loss = ",
                loss1,
                "W=",
                sess.run(W),
                "bias=",
                sess.run(b))
            if not (loss1 == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss1)

    print("Finished")
    print(
        "cost=",
        sess.run(
            loss,
            feed_dict={
                X: train_x,
                Y: train_y}),
        "W=",
        sess.run(W),
        "bias=",
        sess.run(b))

    ##########################################################################
    ## 训练模型可视化                                                                      ##
    ##########################################################################

    plt.plot(train_x, train_y, "ro", label="Original dataset")
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label="yuce")
    plt.legend()
    plt.show()
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs.Training loss")

    plt.show()

    # 训练结束后起动保存权重
    saver.save(sess, "./checkpoint/checkpoint.cpkt")

    # 设置检查点
load_epoch = 198
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())
    saver.save(sess2, "./checkpoint/checkpoint.cpkt-" + str(load_epoch))
    print("x=0.2,z=", sess2.run(z, feed_dict={X: 0.2}))

print("last checkpoints")
print(tf.train.Saver.last_checkpoints)
##########################################################################
## 预测                                                                              ##
##########################################################################

# 查看保存权重内部的数据
saverdir = "./checkpoint/"
print_tensors_in_checkpoint_file(saverdir + "checkpoint.cpkt", None, True)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./checkpoint/checkpoint.cpkt")
    print(sess.run(W))
    print("x=0.2,z=", sess.run(z, feed_dict={X: 0.2}))
