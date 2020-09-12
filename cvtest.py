import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
""" from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config) """

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

model = load_model("model.h5")

capture = cv2.VideoCapture(0)


def predictf(frame):

    im = frame[0:250, 0:250]
    gray = im
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    gray = cv2.resize(gray, (96, 96))

    print(gray.shape)

    res = model.predict_classes(gray.reshape(1, 96, 96, 1))[0]

    answer = res
    return answer


l = []

s = ''

while(True):
    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.rectangle(frame, (0, 0), (250, 250), (0, 255, 0), 3)

    cv2.putText(frame, str(s), (600, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (225, 0, 0), 2, cv2.LINE_AA)

    r, f = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY_INV)

    im = f[0:250, 0:250]
    # print(np.sum(im))
    if np.sum(im) >= 1300000:
        s = predictf(frame)
    else:
        s = ''

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

# After the loop release the cap object
capture.release()
# Destroy all the windows
cv2.destroyAllWindows()
