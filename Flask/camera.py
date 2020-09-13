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

model = load_model("test_model.h5")


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self):
        # extracting frames
        ret, frame = self.video.read()

        frame = cv2.flip(frame, 1)

        roi = frame[0:350, 0:350]
        cv2.rectangle(frame, (0, 0), (350, 350), (0, 255, 0), 2)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(roi, (7, 7), 0)

        gray = cv2.resize(gray, (96, 96))

        res = np.argmax(model.predict(gray.reshape(1, 96, 96, 3)), axis=-1)
        # print(res)

    
        cv2.putText(frame, chr(res[0]+65), (600, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (225, 0, 0), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
