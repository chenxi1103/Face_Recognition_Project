#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Chenxi Li on 2018-12-06
import tensorflow as tf
import cv2
import numpy as np
import os.path
from keras import backend as K
from IntelliCat.face_detection import *

CLASS = 7
CNN_PATH = "IntelliCat/cnnmodel/emotion/cnnmodel"

# Create your tests here.
def analyze_emotion(path):
    image = cv2.imread(path, 1)
    count = 0
    result = []
    for normalized_face, (x, y, w, h) in find_faces_emo(image):
        count = count + 1
        emotion_prediction, logit = emotion_predict(normalized_face)
        print(emotion_prediction)
        if (emotion_prediction == 0):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Angry", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(0)
        elif (emotion_prediction == 1):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Fear",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 0, 0), 2)
            result.append(1)
        elif (emotion_prediction == 2):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Fear",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 0, 0), 2)
            result.append(2)
        elif (emotion_prediction == 3):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Sad",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 0, 0), 2)
            result.append(3)
        elif (emotion_prediction == 4):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Happy",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 0, 0), 2)
            result.append(4)
        elif (emotion_prediction == 5):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Surprise",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 0, 0), 2)
            result.append(5)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Neutral",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 0, 0), 2)
            result.append(6)
    cv2.imwrite(path, image)
    K.clear_session()
    return result, logit

def emotion_predict(face):
    face = face * (1./255)  # normalize
    face = cv2.resize(face, (48*48, 1))
    emotions,logit = predict_cnn(face)
    return emotions, logit

# use cnn model to predict
def predict_cnn(data):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(CNN_PATH + '.meta')
        loader.restore(sess, CNN_PATH)
        x = loaded_graph.get_tensor_by_name('data:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        y = loaded_graph.get_tensor_by_name('probability:0')
        y_ = loaded_graph.get_tensor_by_name('label:0')
        logit = sess.run(y, feed_dict={
            x: data, y_: np.zeros((8, CLASS)), keep_prob: 1.0
        })
        emotions = sess.run(tf.argmax(logit, 1))
    return emotions,logit