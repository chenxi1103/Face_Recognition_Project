#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Chenxi Li on 2018-12-07
import cv2
import numpy as np
import os.path
from face_detection import find_faces
from keras.models import load_model
from keras import backend as K


# Create your tests here.
def analyze_picture(path):
    classifier = load_model('./race_cnn_model.h5')
    result = []
    result_prob = []
    image = cv2.imread(path, 1)
    count = 0
    for normalized_face, (x, y, w, h) in find_faces(image):
        count = count + 1
        race_prediction = classifier.predict_proba(normalized_face, batch_size=32, verbose=0)[0]
        curr = race_prediction.tolist()
        result_prob.append(curr)
        curr_result = curr.index(max(curr))
        print(curr_result)
        if curr_result == 0:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Hispanic", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(0)
        elif curr_result == 1:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Caucasian", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(1)
        elif curr_result == 2:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Asian", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "African", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(3)
    print(result)
    cv2.imwrite(path, image)
    K.clear_session()
    return result, result_prob

if __name__ == '__main__':
    path = "data/sample/"
    file_name = input("Specify image file: ")
    path = path + file_name
    result= analyze_picture(path)

    # prob_result = []
    # real_result = []
    # for i in range(len(result)):
    #     curr = result[i].tolist()
    #     total = sum(curr[0])
    #     temp = []
    #     print(curr)
    #     for j in range(4):
    #         temp.append(curr[0][j] / total * 100)
    #     prob_result.append(temp)
    #     real_result.append(prob_result[i].index(max(prob_result[i])))
    #
    # print(prob_result)
    # print(real_result)