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
from keras.regularizers import l2

import pickle as pkl

import threading
#from multiprocessing import Pool

import random
import time
import pdb


##########################
####### PARAMETERS #######
##########################
CLASSIF_FOLDER = "myclassif/"


##########################
##########################

def folder2labels(path):
    pass

def split_traintest(files):
    pass


def save_img_h5(data, h5_file, img_mean=None, cache_size=1600):
    f = h5py.File(h5_file, "w")
    files = [f for (f,l) in data]
    labels = np.array([l for (f,l) in data])
    n_samples = len(data)

    lab_dset = f.create_dataset("labels", data=labels,compression="gzip")
    
    img_dset = f.create_dataset("imgs", (n_samples,3,256,256),
                                dtype='float32',
                                compression="gzip")
    n_step = len(data)/cache_size
    if n_step % cache_size != 0:
        n_step += 1
    for i in range(n_step):
        X = preprocess_img_batch(files[j*cache_size:(j+1)*cache_size],
                                 img_mean=img_mean)
        img_dset[j*cache_size:(j+1)*cache_size] = X
    f.close()
        
    

def load_img_h5(path):
    pass






def ImageGenerator(data, batch_per_cache=100, batch_size=16, shuffle=False,
                   img_mean=None, **kwargs):
    """
    Generator for the data.
    It splits the data into large caches, and then use keras' ImageDataGenerator
    inside a cache to generate batches

    imgs: list or string
        Contains the data, can be of two forms : 
        * list of (filename, label)
        * str containing the name of the pre-trained h5 file

    cache_size: int
        Number of images pre-loaded

    batch_size: int
        Size of the batch.

    shuffle: bool
        Wether the data must be shuffled.
        It shuffles the order of the caches, and then it shuffles uniformly inside
        the cache
    """
    
    datagen = ImageDataGenerator(kwargs)

    

    preloaded = (type(data) == str)
    n_samples = XXX.shape if preloaded else len(data)


    if preloaded:
        datafile=PDFPZURPBFRIZB
        labels = LJZMJZMRG
    else:
        files = [f for (f,l) in data]
        labels = np.array([l for (f,l) in data])

    n_labels = len(set(labels))
    if n_labels <= 1:
        raise ValueError("Can't learn to classify with only one class")
    elif n_labels == 2:
        Y = labels
    else:
        Y = np_utils.to_categorical(Y, n_labels)
    
    

    n_step = len(data)/(batch_size*batch_per_cache)
    if len(data) % (batch_size*batch_per_cache) != 0:
        n_step += 1

    permutation = np.random.permutation(n_step) if shuffle else np.arange(n_step)
    k = 0
    j = permutation[k]

    if preloaded:
        X_train = lksjfglkjergmzkjer
    else:
        X_train = preprocess_image_batch(files[j*max_n_pic:(j+1)*max_n_pic],
                                         img_mean=img_mean)
    
    Y_train = Y[j*max_n_pic:(j+1)*max_n_pic]
    datagen.fit(X_train)

    while True:
        k = (k + 1)  % n_step
        j = permutation[k]
        out_xtrain = []
        if preloaded:
            moirhmfoiernmg
        else:
            t_xtrain = threading.Thread(target=preprocess_image_batch,
                                        args=(files[j*batch_size*batch_per_cache:
                                                    (j+1)*batch_size*batch_per_cache]),
                                        kwargs={"img_mean":img_mean,
                                                "out":out_xtrain})
        t_xtrain.daemon=True
        t_xtrain.start()
        gen = datagen.flow(X_train,Y_train,
                           batch_size=batch_size,shuffle=shuffle)
        
        for l in range(X_train.shape[0]/batch_size):
            x,y = next(gen)
            yield x,y
            
        t_xtrain.join()
        X_train = out_xtrain[0]
        
        # X_train = preprocess_image_batch2(files_processed[j*max_n_pic:(j+1)*max_n_pic],
        #                                   n_jobs=1)
        # Y_train = Y[j*max_n_pic:(j+1)*max_n_pic]
