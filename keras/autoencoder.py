# -*- coding:utf-8 -*-
# /usr/bin/python
'''
@Author  :  Yan Errol 
@Describe:  
@Evn     :  
@Date    :  2019-07-31  16:00
'''
import os
import numpy as np
from PIL import Image
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
mnist_train_data = "./dataset/mnist/mnist_train/"
def get_data(data_dir):
    file_list = os.listdir(data_dir)
    np.random.shuffle(file_list)
    datas = []
    labels = []
    for file in file_list:
        files = file.split(".")
        if files[-1] == "jpg":
            image = np.array(Image.open(data_dir + file))
            image = image.flatten() # 将28x28的图片转为784为的向量
            image = image / 255.
            datas.append(image)
            labels.append(int(files[0]))
        datas = np.array(datas)
        labels = np.array(labels)
    return datas, labels

x_train, y_train = get_data(mnist_train_data)

# encoder
encoder = keras.Sequential([Dense(256, activation='relu'),Dense(64, activation='relu'),Dense(16, activation='relu'),Dense(2)])
# decoder
decoder = keras.Sequential([Dense(16, activation='relu'),Dense(64, activation='relu'),Dense(256, activation='relu'),Dense(784, activation='sigmoid')])

# autoencoder
autoencoder = keras.Sequential([encoder,decoder])

# compile
autoencoder.compile(optimizer='adam', loss=keras.losses.binary_crossentropy)
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256)
is_reconstruct = False

if is_reconstruct:
    # 图像重构
    examples_to_show = 10
    rec = autoencoder.predict(x_train[:examples_to_show])
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(x_train[i], (28, 28)))
        a[1][i].imshow(np.reshape(rec[i], (28, 28)))
        plt.show()
else:
    # classification
    batch_x = x_train[:1000]
    batch_y = y_train[:1000]
    encoded = encoder.predict(batch_x)
    plt.scatter(encoded[:, 0], encoded[:, 1], c=batch_y)
    plt.show()
