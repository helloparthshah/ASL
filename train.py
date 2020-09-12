import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.core import Activation
from keras import regularizers
import tensorflow as tf
import keras
import os
import random
import cv2


def load_images(directory):
    images = []
    labels = []
    uniq_labels = sorted(os.listdir(directory))

    for idx, label in enumerate(uniq_labels):
        print(label, " is ready to load")
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (96, 96))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)

    # images = images.reshape((len(images), 96, 96, 1))
    print(images.shape)
    images = images.astype('float32')/255
    labels = keras.utils.to_categorical(labels)
    return(images, labels)


images, labels = load_images(directory="./data")
print("Data has been loaded")

print(labels[1])
""" c = list(zip(images, labels))

random.shuffle(c)

images, labels = zip(*c)

x_train = images[0:500]
y_train = labels[0:500]
x_test = images[500:]
y_test = labels[500:] """
# sample = images[600]
# print(sample.shape)
# plt.imshow(sample)
# plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(
        96, 96, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

history = model.fit(images, labels,
                    batch_size=64, epochs=5,
                    verbose=1)

# saving the model
model.save("test_model.h5")

""" plt.imshow(x_test[20])
plt.show()
print(y_test[20])
res = model.predict(x_test[20])
print(res)
 """
