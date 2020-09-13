import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np

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

model = load_model("../model1.h5")


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)
        self.letter = ''

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self):
        # extracting frames
        ret, frame = self.video.read()

        frame = cv2.flip(frame, 1)

        roi = frame[0:250, 0:250]

        # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(roi, (7, 7), 0)

        gray = cv2.resize(gray, (96, 96))

        res = model.predict(gray.reshape(1, 96, 96, 3))

        prediction = np.argmax(res, axis=-1)
        # print(res)

        char = prediction[0]+65
        if char > 80:
            char += 1

        self.letter = chr(char)

        cv2.rectangle(frame, (0, 0), (250, 250), (0, 255, 0), 2)
        cv2.putText(frame, chr(char), (600, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (225, 0, 0), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
