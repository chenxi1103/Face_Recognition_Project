#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Chenxi Li on 2018-12-04

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model


# initialize a CNN
classifier = Sequential()

# create a convolution layer
classifier.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))

# create the maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add second convolution layer / maxpooling layer
classifier.add(Convolution2D(64, (5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# # Add third convolution layer / maxpooling layer
# classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))

# flatten the data
classifier.add(Flatten())

# create the full connection layer
classifier.add(Dense(units= 128, activation='relu'))
classifier.add(Dropout(0.5))

# create the output layer
classifier.add(Dense(units= 1, activation='sigmoid'))
# classifier.add(Dropout(0.5))

# compile CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=80,
                         epochs=100,
                         validation_data=test_set,
                         validation_steps=5)

classifier.save('../gender_drop_cnn_model.h5')
