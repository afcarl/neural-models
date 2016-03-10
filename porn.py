import numpy as np

from os import listdir
from os.path import isfile, join


from convnets import convnet, preprocess_image_batch

mypath='/mnt/data/datasets/flickrdataset/'

reject_folders = ['bikinis', 'lingeries', 'porn', 'swimsuits']
accept_folders = ['sports','women','babies']

reject_files = [f for fold in reject_folders \
                for f in listdir(join(mypath, fold))]
accept_files = [f for fold in accept_folders \
                for f in listdir(join(mypath, fold))]



stop


feat = convnet('vgg_16', weights_path='weights/vgg16_weights.h5', output_layer='dense_2')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
feat.compile(optimizer=sgd, loss='categorical_crossentropy')
    
model = Sequential()
model.add(Dense(2, activation='softmax', name='softmax'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')


