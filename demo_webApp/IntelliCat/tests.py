from django.test import TestCase
import cv2
import numpy as np
import os.path
from .face_detection import find_faces
from keras.models import load_model
from keras import backend as K


# Create your tests here.
def analyze_picture(path):
    classifier = load_model('IntelliCat/gender_cnn_model.h5')
    result = []
    image = cv2.imread(path, 1)
    count = 0
    for normalized_face, (x, y, w, h) in find_faces(image):
        count = count + 1
        gender_prediction = classifier.predict(normalized_face, batch_size=None, verbose=0, steps=None)
        print(gender_prediction)
        if (gender_prediction == 1):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "male", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            result.append(0)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, "female", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            result.append(1)
    cv2.imwrite(path, image)
    K.clear_session()
    return result

def put_data(path, gender):
    image = cv2.imread(path, 1)
    [dir, file] = os.path.split(path)
    dir = dir + '/' + gender + '/'
    path = dir + file
    print(path)
    cv2.imwrite(path, image)
    print("success")
