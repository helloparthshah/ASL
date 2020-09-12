import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

model = load_model("test_model.h5")

camera = cv2.VideoCapture(0)

while True:
    (t, frame) = camera.read()
    frame = cv2.flip(frame, 1)

    roi = frame[0:250, 0:250]
    cv2.rectangle(frame, (0, 0), (250, 250), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(roi, (7, 7), 0)

    gray = cv2.resize(gray, (96, 96))

    res = model.predict_classes(gray.reshape(1, 96, 96, 3))

    print(chr(res[0]+65))

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    cv2.imshow('frame', frame)

    keypress = cv2.waitKey(1)

    if keypress == 27:
        break

camera.release()
cv2.destroyAllWindows()
