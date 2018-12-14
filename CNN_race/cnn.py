#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Chenxi Li on 2018-12-07
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# imports needed for CNN
import csv
import os, glob
import pandas as pd
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


def createCNNModel():
    model = Sequential()
    # create a convolution layer
    model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    # model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    # create the maxpooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add second convolution layer / maxpooling layer
    model.add(Convolution2D(64, (5, 5), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add third convolution layer / maxpooling layer
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flatten the data
    model.add(Flatten())

    # create the full connection layer
    model.add(Dense(units=32, activation='relu'))
    # create the output layer
    model.add(Dense(units=3, activation='relu'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

model = createCNNModel()
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=25,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=25,
                                            class_mode='categorical')

model.fit_generator(training_set,
                    steps_per_epoch=35,
                    epochs=50,
                    validation_data=test_set,
                    validation_steps=2.4)

model.save('../raceclean_cnn_model.h5')
