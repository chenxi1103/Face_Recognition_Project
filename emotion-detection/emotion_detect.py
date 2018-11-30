import cv2
import model

IMG_PATH = "Asian/AF0301_1100_30R.jpg"
HAAR_PATH = "/usr/local/Cellar/opencv/3.4.3/share/OpenCV/haarcascades"
HAAR_FACE_PATH = HAAR_PATH + "/haarcascade_frontalface_default.xml"
emotion_labels = ['angry', "disgust", 'fear', 'happy', 'sad', 'surprise', 'neutral']


def face_detect(image):
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

    emotions = model.predict_linear(face)

    return emotions


if __name__ == '__main__':

    img = cv2.imread(IMG_PATH)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    face_rect = face_detect(img_eq)  # rectangle region of the detected face
    for (x, y, w, h) in face_rect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    emotions = emotion_predict(img_eq)
    for emotion in emotions:
        print(emotion_labels[emotion])

    #cv2.imshow("face detected image", img)

    #cv2.waitKey()
    #cv2.destroyAllWindows()
