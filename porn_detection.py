import sys

sys.path.insert(0, "/home/lblier/.local/lib/python2.7/site-packages")

import numpy as np

from os import listdir
from os.path import isfile, join


from convnets import convnet, preprocess_image_batch, preprocess_image_batch2
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import pickle as pkl

import random


model = convnet('alexnet', weights_path='weights/alexnet_weights.h5',
                output_layer='dense_2')
model.add(Dense(1,
                activation='sigmoid',
                name='classifier'))
model.load_weights("porn_weights.h5")

sgd = SGD(lr=.5, decay=1.e-6, momentum=0., nesterov=False)
model.compile(optimizer=sgd, loss='binary_crossentropy')




img_paths = []
X = preprocess_image_batch2(img_paths)

y = model.predict(X)



