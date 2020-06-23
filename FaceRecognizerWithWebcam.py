
from PIL import Image

import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model(r'faceRecognizer.h5')

# Loading the cascades, a
face_cascade = cv2.CascadeClassifier('Your haracascade file location')


def face_extractor(img):
    # a list containing features of all the faces in the frame
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]

    return roi


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    success, frame = video_capture.read()

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)

        name = "None matching"
        # i is  number of different classes in your dataset
        if (pred[0][i] > 0.5):

            name = 'ith name'

        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()