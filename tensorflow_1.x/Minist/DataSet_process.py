#!/usr/bin/python
# -*-coding:utf-8-*-
# Author:YanErrol
# Date:2019.04.09

import pylab
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("dataset/", one_hot=True)

print("输入数据：", mnist.train.images)
print("输入数据打shape：", mnist.train.images.shape)


tf.reset_default_graph()
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

print("x=", x, "y=", y)

W = tf.Variable(tf.random_normal(([784, 10])))
b = tf.Variable(tf.zeros([10]))

print(W, b)

pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数
loss = tf.reduce_mean(-1 * tf.reduce_sum(y *
                                         tf.log(pred), reduction_indices=1))

# 定义参数
learning_rate = 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


training_epochs = 25
batch_size = 100
display_step = 1

# 定义保存权重器
saver = tf.train.Saver()

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # 循环所有数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # 运行优化器
            _, cost = sess.run([optimizer, loss], feed_dict={
                               x: batch_xs, y: batch_ys})

            # 计算平均loss值
            avg_cost += cost / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:", "%04d" % (epoch + 1), "loss=", "%04f" % avg_cost)

    print("Finished!")


# 测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(
        {x: mnist.test.images, y: mnist.test.labels}))

    model_path = "./checkpoint/checkpoint_dataset.cpkt"
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)


# 读取模型
print("Starting and session ...")
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 恢复模型变量
    saver.restore(sess, model_path)

    # 测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(
        {x: mnist.test.images, y: mnist.test.labels}))

    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={x: batch_xs})

    print(outputval, predv, batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
