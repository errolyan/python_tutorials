#coding:utf-8
# @sinner
# 2016/05/01

from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import os
import numpy as np
from PIL import Image


def load_data(training):

    data = None
    label = None
    filedir = ''

    if training:
        filedir = './mnist_train'
    else:
        filedir = './mnist_test'
    filelist = os.listdir(filedir)
    imagelist = []
    for filename in filelist:
        splits = str(filename).split('.')
        if splits[-1] == 'jpg':
            imagelist.append(filename)

    imageCount = len(imagelist)
    data = np.empty((imageCount, 1, 28, 28), dtype="float32")
    label = np.empty((imageCount,), dtype="uint8")
    j = 0
    for filename in imagelist:
        splits = str(filename).split('.')
        img = Image.open(filedir + "/" + filename)
        arr = np.asarray(img, dtype='float32')
        data[j, :, :, :] = arr
        label[j] = int(filename.split('.')[0])
        j += 1
    return data, label


def getModel():
    #construct cnn model
    model = Sequential()

    #first layer, 6 con cores
    model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #second layer, 8 con cores
    model.add(Convolution2D(16, 5, 5, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #first full connection layer
    model.add(Flatten())
    model.add(Dense(output_dim=120))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_dim=84))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    #Softmax output layer, 10 classes
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def train(model):
    data, label = load_data(True)
    data = data.reshape(data.shape[0], 1, 28, 28)
    data = data.astype('float32')
    data /= 255
    label = np_utils.to_categorical(label, 10)
    #begin to train
    model.fit(data, label, batch_size=128, nb_epoch=10, shuffle=True, verbose=1, validation_split=0.05)
    model.save_weights('Lenet_.hdf5', overwrite=True)


def test(model):
    data, label = load_data(False)
    data = data.reshape(data.shape[0], 1, 28, 28)
    data = data.astype('float32')
    data /= 255
    model.load_weights('Lenet.hdf5')
    outcome = model.predict_classes(data, batch_size=32, verbose=1)
    count = 0
    for x in xrange(0, len(label)):
        if outcome[x] == label[x]:
            count += 1
    print('count = ' + str(count) + ', acc = ' + str(float(count) / len(label)))


# train(getModel())
test(getModel())
