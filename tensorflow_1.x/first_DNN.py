# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  
@Evn     :  
@Date    :   - 
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 批大小
BATCH_SIZE = 8
# 学习率
learning_rate = 0.001
# 训练轮次
epochs = 30000

# 设置随机种子
SEED = 23455

# 0.产生数据集
rdm = np.random.mtrand.RandomState(SEED)
print(rdm)

# 随机数返回32行2列的矩阵 表示32组 体积和重量 作为输入数据集
X = rdm.rand(32, 2)
print('X =',X)
# 从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0
# 作为输入数据集的标签（正确答案）
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print('Y_ =',Y_)

# 1.定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
W2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

hidden_layer_input = tf.matmul(x,W1)
y = tf.matmul(hidden_layer_input, W2)

# 2.定义反向传播
loss_mse = tf.reduce_mean(tf.square(y-y_))

## 梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_mse)

# 定义保存权重器
saver = tf.train.Saver()


# 3.生成会话，训练数据
with tf.Session() as sess:
    with tf.device("cpu:0"):
        # 初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 输出初始化值的权重
        print("W1:\n", sess.run(W1))
        print("W2:\n", sess.run(W2))
        print("\n")

        # 训练3000
        for i in range(epochs):
            start = (i*BATCH_SIZE) % 32
            end = start + BATCH_SIZE
            # train_step梯度下降法训练数据，feed_dict为喂入特征数据
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})

            # 每500轮打印loss值, 便于观察损失函数降低的过程
            if i % 50 == 0:
                total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
                print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))

        # 输出训练后的参数取值。
        print("\n")
        print("W1:\n", sess.run(W1))
        print("W2:\n", sess.run(W2))
        saver.save(sess, "./checkpoint/checkpoint11.cpkt")

# 预测
with tf.Session() as sess1:
    saver = tf.train.Saver()
    sess1.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./checkpoint/checkpoint11.cpkt.meta')
    saver.restore(sess1, "./checkpoint/checkpoint11.cpkt")
    print("Restore is over!")
    start = (i * BATCH_SIZE) % 32
    end = start + BATCH_SIZE
    x_input = [[ 0.83494319,0.11482951]]
    # 预测
    y = sess1.run(y, feed_dict={x: x_input})
    print(y)