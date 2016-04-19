import sys

sys.path.insert(0, "/home/lblier/.local/lib/python2.7/site-packages")

import numpy as np
from os import listdir
from os.path import isfile, join

import h5py

from convnets import convnet, preprocess_image_batch, preprocess_image_batch2
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from sklearn.cross_validation import StratifiedShuffleSplit

import pickle as pkl

from threading import Thread
#from multiprocessing import Pool

import random
import time
import pdb

import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')

##########################
####### PARAMETERS #######
##########################
CLASSIF_FOLDER = "myclassif/"


##########################
##########################

def folder2labels(path):
    #TODO
    pass


def split_traintest(data):
    sss = StratifiedShuffleSplit([l for f,l in data])
    for train_index, test_index in sss:
        data_train = [data[i] for i in train_index]
        data_test  = [data[i] for i in  test_index]
    return data_train, data_test



def save_img_h5(data, h5_file, cache_size=1600):
    f = h5py.File(h5_file, "w")
    files = [name for (name,l) in data]
    labels = np.array([l for (name,l) in data])
    n_samples = len(data)

    lab_dset = f.create_dataset("labels", data=labels,compression="gzip")
    
    img_dset = f.create_dataset("imgs", (n_samples,3,256,256),
                                dtype='float32',
                                compression="gzip")
    n_step = len(data)/cache_size
    if n_step % cache_size != 0:
        n_step += 1
    for j in range(n_step):
        print(str(j)+"/"+str(n_step))
        X = preprocess_image_batch(files[j*cache_size:(j+1)*cache_size])
        img_dset[j*cache_size:(j+1)*cache_size] = X
    f.close()

def preprocessing(data, folder_path):
    data_train, data_test = split_traintest(data)
    train_files = join(folder_path, "train_files.pkl")
    test_files = join(folder_path, "test_files.pkl")
    train_set = join(folder_path,"train_set.h5")
    test_set = join(folder_path,"test_set.h5")
    pkl.dump(data_train, open(train_files,"wb"))
    pkl.dump(data_test, open(test_files, "wb"))
    save_img_h5(data_train, train_set)
    save_img_h5(data_test, test_set)
    return train_files, test_files,train_set, test_set
        
    

def load_img_h5(h5dataset,start,stop,out=None):
    data = h5dataset[start:stop]
    logging.debug("load_img_h5, data shape : "+str(data.shape))
    if not out is None:
        out.append(data)
        logging.debug("len out : "+str(len(out)))
        
    else:
        return data




def ImageGenerator(data, img_size, batch_per_cache=100, batch_size=16,
                   shuffle=False,**kwargs):
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
    cache_size = batch_size*batch_per_cache

    if preloaded:
        f = h5py.File(data, "r")
        imgs,labels = f["imgs"],f["labels"]
        n_samples = labels.shape[0]
    else:
        files = [f for (f,l) in data]
        labels = np.array([l for (f,l) in data])
        n_samples = len(data)

    logging.debug("Data loaded")
    logging.debug("Data shape : "+str(imgs.shape))

    n_labels = len(set(labels))
    if n_labels <= 1:
        raise ValueError("Can't learn to classify with only one class")
    elif n_labels == 2:
        Y = labels
    else:
        Y = np_utils.to_categorical(Y, n_labels)
    
    n_step = len(data)/cache_size
    if len(data) % cache_size != 0:
        n_step += 1

    permutation = np.random.permutation(n_step) if shuffle else np.arange(n_step)
    k = 0
    j = permutation[k]

    start,stop = j*cache_size,(j+1)*cache_size
    logging.debug("Loading cache ...")
    logging.debug("Cache index : "+str((start, stop)))
    if preloaded:
        X_cache = load_img_h5(imgs,start,stop)
    else:
        X_cache = preprocess_image_batch(files[start,stop])

    logging.debug("Cache loaded")
    logging.debug("Cache shape : "+str(X_cache.shape))
    
    Y_cache = Y[start:stop]
    datagen.fit(X_cache)

    while True:
        logging.debug("Beginning the loop")
        logging.debug("Cache shape : "+str(X_cache.shape))
        logging.debug(str((X_cache.shape, Y_cache.shape)))
        k = (k + 1)  % n_step
        j = permutation[k]
        out_xcache = []
        start,stop = j*cache_size,(j+1)*cache_size
        if preloaded:
            t_xcache = Thread(target=load_img_h5,
                              args=(imgs,start,stop,out_xcache))
                              #kwargs={"out":out_xcache})
        else:
            t_xcache = Thread(target=preprocess_image_batch,
                              args=(files[start:stop], out_xcache))
                              #kwargs={"out":out_xcache})
        t_xcache.daemon=True
        t_xcache.start()
        gen = datagen.flow(X_cache,Y_cache,
                           batch_size=batch_size,shuffle=shuffle)
        
        for l in range(X_cache.shape[0]/batch_size):
            
            
            x,y = next(gen)
            mean1,mean2 = x.shape[2], x.shape[3]
            yield x[:,:,(mean1-img_size)/2:(mean1+img_size)/2,
                    (mean2-img_size)/2:(mean2+img_size)/2], y
            
        t_xcache.join()
        X_cache = out_xcache[0]
        Y_cache = Y[j*cache_size:(j+1)*cache_size]
