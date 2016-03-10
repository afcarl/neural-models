import numpy as np

from os import listdir
from os.path import isfile, join


from convnets import convnet, preprocess_image_batch
from keras.utils import np_utils
import random

mypath='/mnt/data/datasets/flickrdataset/'

reject_folders = ['bikinis', 'lingeries', 'porn', 'swimsuits']
accept_folders = ['sports','women','babies']

reject_files = [(f,0) for fold in reject_folders \
                for f in listdir(join(mypath, fold))]
accept_files = [(f,1) for fold in accept_folders \
                for f in listdir(join(mypath, fold))]



data = reject_files + accept_files
random.shuffle(data)


data_processed = []
feat = convnet('vgg_16', weights_path='weights/vgg16_weights.h5', output_layer='dense_2')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
feat.compile(optimizer=sgd, loss='categorical_crossentropy')
batch_size = 16
for i in range(0,len(data), batch_size):
    print "Processing images "+(i, i+batch_size)
    files_processed = [f for (f,l) in data[i:i+batch_size]]
    img_batch = preprocess_image_batch(files_processed)
    features_batch = feat.predict(img_batch)
    data_processed.extend(features_batch)
    


    
train = data[:10604]
valid = data[10604:11604]
test = data[11604:]





    
model = Sequential()
model.add(Dense(2, activation='softmax', name='softmax'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


