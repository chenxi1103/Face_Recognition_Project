import cv2
import numpy as np
faceCascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_alt.xml')

def find_faces(image):
    coordinates = locate_faces(image)
    cropped_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in coordinates]
    normalized_faces = [normalize_face(face) for face in cropped_faces]
    return zip(normalized_faces, coordinates)

def find_faces_emo(image):
    coordinates = locate_faces(image)
    cropped_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in coordinates]
    normalized_faces = [normalize_face_emo(face) for face in cropped_faces]
    return zip(normalized_faces, coordinates)

def normalize_face_emo(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    return face;

def normalize_face(face):
    face = cv2.resize(face, (64, 64),interpolation = cv2.INTER_CUBIC)
    # face = np.array([face])
    face = np.expand_dims(face, axis=0)
    return face
    # im = cv2.resize(face, (64, 64))
    # im.reshape((64, 64))
    # batch = np.expand_dims(im, axis=0)
    # batch = np.expand_dims(batch, axis=3)
    # return batch


def locate_faces(image):
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces