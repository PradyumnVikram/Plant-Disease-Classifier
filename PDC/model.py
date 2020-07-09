# the libraries required
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Flatten, Dropout
import tensorflow as tf
import pickle
import random
from tqdm import tqdm
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
print('start')
root = os.path.dirname(__file__)
DIR = os.path.join(root, 'dataset', 'plantvillageog', 'plantvillage dataset', 'color')
class_num = []
training_data = []
CATEGORIES = []
X = []
y = []
IMG_SIZE = 75

for directory in os.listdir(DIR):
    CATEGORIES.append(directory)


def create_data():
    for c in CATEGORIES:
        PATH = os.path.join(DIR, c)
        class_num = CATEGORIES.index(c)
        for img in os.listdir(PATH):
            IMGDIR = os.path.join(PATH, img)
            if os.path.isfile(IMGDIR):
                img_array = cv2.imread(IMGDIR, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            else:
                print('error')


create_data()

for features, labels in training_data:
    X.append(features)
    y.append(labels)

NAME = 'Model1.7'


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X/255.0
y = tf.keras.utils.to_categorical(y, 38)

# the convolutional neural network

clf = Sequential()
print(X.shape[1:])
clf.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
clf.add(Activation('tanh'))
clf.add(MaxPooling2D(pool_size=(2, 2)))

clf.add(Conv2D(64, (3, 3)))
clf.add(Activation('relu'))
clf.add(MaxPooling2D(pool_size=(2, 2)))

clf.add(Flatten())
clf.add(Activation('relu'))
clf.add(Dropout(0.2))

clf.add(Dense(38))
clf.add(Activation('softmax'))

tensorboard = TensorBoard(logdir=f'logs/{NAME}')

clf.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
            callbacks=[tensorboard])

clf.fit(X, y, batch_size=5,
        epochs=5,
        validation_split=0.3)

clf.save('Seq_Acc-9185_Loss-2670_ValAcc-7738_ValLoss-7729__cat_33.model')
