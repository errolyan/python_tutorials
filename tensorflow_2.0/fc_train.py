# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Errol 
@Describe:  全连接神经网络
@Evn     :  
@Date    :   - 
'''
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():
    '''
    数据集加载和批次分配
    :return:
    '''
    (x, y), _ = datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

@tf.function
def prepare_mnist_features_and_labels(x, y):
    '''
    处理数据特征和标签
    :param x:
    :param y:
    :return:
    '''
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


'''
搭建模型
'''
model = keras.Sequential([
    layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.Adam()


@tf.function
def compute_loss(logits, labels):
    '''
    计算损失函数
    :param logits:
    :param labels:
    :return:
    '''
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
    '''计算准确率'''
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

@tf.function
def train_one_step(model, optimizer, x, y):
    '''
    计算一步的迭代
    :param model:
    :param optimizer:
    :param x:
    :param y:
    :return:
    '''
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    # update to weights
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(logits, y)

    # loss and accuracy is scalar tensor
    return loss, accuracy


def train(epoch, model, optimizer):
    '''
    训练
    :param epoch: 批次
    :param model: 模型
    :param optimizer: 优化器
    :return:
    '''
    train_ds = mnist_dataset()
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(train_ds):
        loss, accuracy = train_one_step(model, optimizer, x, y)

    if step % 500 == 0:
        print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

    return loss, accuracy


def main(epochs_num):
    '''
    主函数
    :param epochs_num:
    :return:
    '''
    for epoch in range(epochs_num):
        loss, accuracy = train(epoch, model, optimizer)

    print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

if __name__=="__main__":
    epochs_num = 20
    main(epochs_num)
