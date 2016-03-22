import sys

sys.path.insert(0, "/home/lblier/.local/lib/python2.7/site-packages")

import numpy as np

from os import listdir
from os.path import isfile, join


from convnets import convnet, preprocess_image_batch
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


import pickle as pkl

import random

mypath='/mnt/data/datasets/flickrdataset/'

labels={'bikinis':0, 'lingeries':1, 'porn':2, 'swimsuits':3,'sports':4,'women':5,'babies':6}
reject_folders = ['bikinis', 'lingeries', 'porn', 'swimsuits']
accept_folders = ['sports','women','babies']

folders = ['bikinis','lingeries','porn','swimsuits','sports','women','babies']


files = [(join(mypath,fold,f),labels[fold]) for fold in folders \
         for f in listdir(join(mypath, fold)) if f[-3:].lower() == "jpg"]

data = files

data_processed = []
feat = convnet('vgg_16', weights_path='weights/vgg16_weights.h5', output_layer='dense_2')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
feat.compile(optimizer=sgd, loss='categorical_crossentropy')
batch_size = 16
for i in range(0,len(data), batch_size):
    print "Processing images "+str((i, i+batch_size))
    files_processed = [f for (f,l) in data[i:i+batch_size]]
    img_batch = preprocess_image_batch(files_processed)
    features_batch = feat.predict(img_batch)
    data_processed.extend(features_batch)

Y = [l for (f,l) in data]

with open('/mnt/data/datasets/flickrdataset/processed.pkl', 'wb') as f:
    pkl.dump((data_processed,Y), f)
    
with open('/mnt/data/datasets/flickrdataset/processed.pkl', 'rb') as f:
    X, y = pkl.load(f)


X = np.stack(X)
X_train, y_train = X[:10604], y[:10604]
X_test, y_test = X[10604:], y[10604:]

Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)


    
model = Sequential()
model.add(Dense(2, activation='softmax', name='softmax',input_shape=(4096,)))
sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

nb_epoch = 100
batch_size=16
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
