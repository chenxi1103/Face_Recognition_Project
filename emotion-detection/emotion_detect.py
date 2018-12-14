# -- Author: Jiayue Bao --
# -- Created Date: 2018/11/27
# -- Output the probability of emotions for an input human face 
# -- used CNN model
import cv2
import tensorflow as tf
import numpy as np
import time


CNN_PATH = "model/cnnmodel"
IMG_PATH = "cry.jpg"
HAAR_PATH = "/usr/local/Cellar/opencv/3.4.3/share/OpenCV/haarcascades"
HAAR_FACE_PATH = HAAR_PATH + "/haarcascade_frontalface_default.xml"
emotion_labels = ['angry', "disgust", 'fear', 'happy', 'sad', 'surprise', 'neutral']


def face_detect(img):
    face_cascade = cv2.CascadeClassifier(HAAR_FACE_PATH)
    faces = face_cascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # rectangle region of detected face
    return faces


def emotion_predict(face):
    face = face * (1./255)  # normalize
    face = cv2.resize(face, (48*48, 1))

    logit = predict_cnn(face)

    return logit


# use cnn model to predict
def predict_cnn(data):
    loaded_graph = tf.Graph()
    startTime = time.time()
    with tf.Session(graph=loaded_graph) as sess:
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)
        print("Time taken: %f" % (time.time() - startTime))
        loader = tf.train.import_meta_graph(CNN_PATH + '.meta')
        print("Time taken: %f" % (time.time() - startTime))
        loader.restore(sess, CNN_PATH)
        print("Time taken: %f" % (time.time() - startTime))
        x = loaded_graph.get_tensor_by_name('data:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        y = loaded_graph.get_tensor_by_name('probability:0')
        y_ = loaded_graph.get_tensor_by_name('label:0')
        print("Time taken: %f" % (time.time() - startTime))
        logit = sess.run(y, feed_dict={
            x: data, y_: np.zeros((8, 7)), keep_prob: 1.0
        })
        #emotions = sess.run(tf.argmax(logit, 1))

    print("Time taken: %f" % (time.time() - startTime))
    sess.close()
    return logit

if __name__ == '__main__':
    img = cv2.imread(IMG_PATH)
    # image preprocessing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    face_rect = face_detect(img_eq)  # rectangle region of the detected face

    for (x, y, w, h) in face_rect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = img_eq[x:x+w, y:y+h]

    logits = emotion_predict(img_eq)
    for i in range(7):
        print(str(emotion_labels[i]) + ": " + str(logits[0][i]))


    #cv2.imshow("face detected image", img)


    #cv2.imshow("face detected image", face_roi)
    #cv2.waitKey()
    #cv2.destroyAllWindows()