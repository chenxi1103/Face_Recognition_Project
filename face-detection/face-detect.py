#!/usr/bin/env python
# coding:utf-8
# ------Author:Chenxi Li--------

import cv2
# the path of target image for face detection
imagePath = "/Users/ChenxiLi/Desktop/4.jpeg"
# Please find the xml files in "opencv_haarcascades" folder and specify your path
facedetect = "/Users/ChenxiLi/Desktop/face-detection/opencv_haarcascades/haarcascade_frontalface_alt2.xml"

# Create the haarcascade clasifier
faceCascade = cv2.CascadeClassifier(facedetect)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = faceCascade.detectMultiScale(gray, 1.3,5)

# Draw a rectangle around the faces
for (x, y, w ,h) in faces:
    img = cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0) , 2)

cv2.imshow("Faces  found" ,image)
cv2.imwrite("/Users/ChenxiLi/Desktop/facefound4.jpeg",image)
cv2.waitKey(0)