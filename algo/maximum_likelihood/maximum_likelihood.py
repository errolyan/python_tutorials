# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author:  Yan Errol  @Email:2681506@gmail.com   
@Date:  2019-06-03  16:12
@File：
@Describe:
@Evn:
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15,10)

N = 100
a = tf.placeholder("float")
b = tf.placeholder("float")
c = tf.placeholder("float")
d = tf.placeholder("float")
x = tf.linspace(-5.0, 10.0, num = N, name = "x")
noise = tf.random_normal(shape = [N,], mean = 0, stddev = 50.0, seed = 1)

# y = a*x^3 + b*x^2 + c*x +d
with tf.Session() as sess:
    y = sess.run(tf.add(tf.add(tf.add(tf.add(tf.multiply(a,tf.pow(x,3)),tf.multiply(b,tf.pow(x, 2))),tf.multiply(c,x)),d),
                        noise), feed_dict= {a: -1, b: 10, c: 1, d: 1} )
    x = np.reshape(np.array(x.eval()), (N,1))
    y = np.reshape(np.array(y) / np.max(y), (N,1))

fig = plt.figure(figsize=(12, 8))
plt.scatter(x, y)
plt.xlabel("x",fontsize = 40)
plt.ylabel("y",fontsize = 40, rotation = 0)
plt.savefig("./data_show.png")
plt.show(0)

learning_rate = 0.1
training_epochs = 1000
display_step = 500

n_samples = x.shape[0]  # number of samples, equal N
n_input = 1  # number of input neurons

for my_num_hidden, my_subplot_num in zip([1, 2, 4, 16], range(1, 5, 1)):

    n_hidden = my_num_hidden  # number of hidden neurons
    print('Fitting curve with {} hidden neurons Maximum Likelihood Neural Network'.format(my_num_hidden))

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    weights = {'w_hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
               'w_output': tf.Variable(tf.random_normal([n_hidden, n_input]))}
    biases = {'b_hidden': tf.Variable(tf.random_normal([n_hidden])),
              'b_output': tf.Variable(tf.random_normal([n_input]))}

    hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['w_hidden']), biases['b_hidden']))
    output_layer = tf.add(tf.matmul(hidden_layer, weights['w_output']), biases['b_output'])

    cost = tf.reduce_mean(tf.pow(output_layer - Y, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    my_cost = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            c, _ = sess.run([cost, optimizer], feed_dict={X: x, Y: y})
            my_cost.append(c)
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost = ", "{:.9f}".format(c))
        pred = sess.run(output_layer, feed_dict={X: x})
    plt.subplot(int("22" + str(my_subplot_num)))
    plt.scatter(x, y)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20, rotation=0)
    plt.plot(x, pred, c='r')
    plt.title('{} hidden neurons'.format(my_num_hidden))
    print('************************************************************')
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.plot(range(training_epochs), my_cost)
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Loss", fontsize=20, rotation=1)
plt.show()