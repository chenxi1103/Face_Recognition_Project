#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Chenxi Li on 2018-12-07
from django.test import TestCase
import cv2
import numpy as np
import os.path
from .face_detection import find_faces
from keras.models import load_model
from keras import backend as K
from keras.models import Sequential


# Create your tests here.
def analyze_race(path):
    classifier = load_model('IntelliCat/race_cnn_model.h5')
    result = []
    result_prob = []
    image = cv2.imread(path, 1)
    count = 0
    for normalized_face, (x, y, w, h) in find_faces(image):
        count = count + 1
        race_prediction = classifier.predict_proba(normalized_face, batch_size=32, verbose=0)[0]
        race_prediction1 = classifier.predict_classes(normalized_face,batch_size=32, verbose=0)
        print(race_prediction1)
        curr = race_prediction.tolist()
        print(curr)
        result_prob.append(avg(curr))
        curr_result = curr.index(max(curr))
        print(curr_result)
        # if curr_result == 0:
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     cv2.putText(image, "Hispanic", (x, y - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #     result.append(0)
        if curr_result == 0:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Caucasian", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(1)
        elif curr_result == 1:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Asian", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "African", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(3)
    cv2.imwrite(path, image)
    K.clear_session()
    return result, result_prob

def avg(curr):
    result = []
    total = sum(curr)
    for i in range(3):
        result.append(curr[i]/total * 100)
    return result
