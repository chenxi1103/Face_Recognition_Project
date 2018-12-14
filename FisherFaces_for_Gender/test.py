#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Chenxi Li on 2018-12-05
#!/usr/bin/env python
# coding:utf-8
# ------Author:Chenxi Li--------
import cv2
import numpy as np
import os.path

from cv2 import WINDOW_NORMAL
from face_detection import find_faces
from PCA import *
from LDA import *

def analyze_picture(eig_v, file):
    image = cv2.imread(file,0)
    imgVector = reshape(image, (1, 128 * 128))
    test = imgVector * eig_v
    testMale = np.load("./trained_result/225/male.npy")
    testFemale = np.load("./trained_result/225/female.npy")
    w = np.load("./trained_result/225/lda_vector.npy")
    gender_prediction = fitLDA(test, w, testMale, testFemale)
    print(gender_prediction)
    return gender_prediction

if __name__ == '__main__':
    eig_v = start_lda()
    gender = input("Please specify gender").lower()
    # dataDir = "/Users/ChenxiLi/Desktop/Face_Recognition_Project/data/test/" + gender
    # files = os.listdir(dataDir)
    files = glob.glob("../data/test/%s/*" % gender)
    random.shuffle(files)
    count = 0
    total = 0
    for file in files:
        # if not os.path.isdir(file):
            predict = analyze_picture(eig_v, file)
            print(predict)
            if gender == 'male' and predict == 0:
                count = count + 1
            elif gender == 'female' and predict == 1:
                count = count + 1
            total = total + 1

    print(count)
    print(total)
    acc = count / total
    print(acc)
