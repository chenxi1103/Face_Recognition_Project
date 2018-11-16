#!/usr/bin/env python
# coding:utf-8
# ------Author:Chenxi Li--------
import cv2
import numpy as np
import os.path

from cv2 import WINDOW_NORMAL
from face_detection import find_faces
from src.PCA import *
from src.LDA import *

ESC = 27

def start_webcam(eig_v, window_size, window_name='live', update_time=50):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    video_feed = cv2.VideoCapture(0)
    video_feed.set(3, width)
    video_feed.set(4, height)
    read_value, webcam_image = video_feed.read()

    delay = 0
    init = True
    while read_value:
        read_value, webcam_image = video_feed.read()
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
          if init or delay == 0:
            init = False
            imgVector = reshape(normalized_face, (1, 350 * 350))
            test = imgVector * eig_v
            testMale = np.load("./trained_result/225/male.npy")
            testFemale = np.load("./trained_result/225/female.npy")
            w = np.load("./trained_result/225/lda_vector.npy")
            gender_prediction = fitLDA(test, w, testMale, testFemale)
          if (gender_prediction == 0):
              #cv2.rectangle(webcam_image, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(webcam_image, "male",
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                          (255, 0, 0), 2)
          else:
            cv2.putText(webcam_image, "female",
                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 255), 2)
        delay += 1
        delay %= 20
        cv2.imshow(window_name, webcam_image)
        key = cv2.waitKey(update_time)
        if key == ESC:
            break

def analyze_picture(eig_v, path, window_size, window_name='static'):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    image = cv2.imread(path, 1)
    for normalized_face, (x, y, w, h) in find_faces(image):
        imgVector = reshape(normalized_face, (1, 350 * 350))
        eig_v = start_lda()
        test = imgVector * eig_v
        testMale = np.load("./trained_result/225/male.npy")
        testFemale = np.load("./trained_result/225/female.npy")
        w = np.load("./trained_result/225/lda_vector.npy")
        gender_prediction = fitLDA(test, w, testMale, testFemale)
        if (gender_prediction == 0):
            #cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(image, "male", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
           # cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(image, "female", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    if key == ESC:
        cv2.destroyWindow(window_name)

if __name__ == '__main__':
    eig_v = start_lda()
    choice = input("Welcome to 18794 Gender Detection, Use Camera?(y/n) ").lower()
    if (choice == 'y'):
        window_name = "18794 Final Project Gender Cam (press ESC to exit)"
        start_webcam(eig_v, window_size=(1280, 720), window_name=window_name,
                     update_time=15)
    else:
        run_loop = True
        window_name = "18794 Final Project Test Result (press ESC to exit)"
        print("Default path is set to data/sample/")
        print("Type q or quit to end program")
        while run_loop:
            path = "../data/sample/"
            file_name = input("Specify image file: ")
            if file_name == "q" or file_name == "quit":
                run_loop = False
            else:
                path += file_name
                if os.path.isfile(path):
                    analyze_picture(eig_v, path, window_size=(1280, 720), window_name=window_name)
                else:
                    print("File not found!")