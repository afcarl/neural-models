#!/usr/bin/env python2
import numpy as np

import argparse
from os.path import join, splitext, basename
import h5py
import imp
from threading import Thread

from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


import ipdb
    

def load_img_h5(h5dataset,start,stop,out=None):
    data = h5dataset[start:stop]
    if not out is None:
        out.append(data)        
    else:
        return data

def load_img_csv(csvdataset, start, stop, out=None):
    pass

def load_labels_h5(h5dataset):
    pass

def load_labels_csv(csvdataset):
    pass


        
    
def ImageGenerator(data, img_size, batch_per_cache=100, batch_size=16,
                   shuffle=False, mode="h5", **kwargs):
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

    mode: h5 or csv
    """
    datagen = ImageDataGenerator(kwargs)

    cache_size = batch_size*batch_per_cache

    if mode == "h5":
        f = h5py.File(data, "r")
        imgs,labels = f["imgs"], f["labels"]
        n_samples = labels.shape[0]

    elif mode == "csv":
        pass

    
    n_labels = len(set(labels))
    Y = np_utils.to_categorical(labels, n_labels)
    count_label = [0 for _ in range(n_labels)]
    for l in labels:
        count_label[l] += 1
    
    n_step = n_samples/cache_size
    if n_samples % cache_size != 0:
        n_step += 1

    permutation = np.random.permutation(n_step) if shuffle else np.arange(n_step)
    k = 0
    j = permutation[k]
    start,stop = j*cache_size,(j+1)*cache_size
    X_cache = load_img_h5(imgs,start,stop)
    Y_cache = Y[start:stop]

    
    
    datagen.fit(X_cache)
    if n_samples < cache_size:
        gen = datagen.flow(X_cache, Y_cache, batch_size, shuffle=shuffle)
        while True:
            x, y = next(gen)
            mean1,mean2 = x.shape[2], x.shape[3]
            yield x[:,:,(mean1-img_size)/2:(mean1+img_size)/2,
                    (mean2-img_size)/2:(mean2+img_size)/2], y
            
            

    while True:
        k = (k + 1)  % n_step
        j = permutation[k]
        out_xcache = []
        start,stop = j*cache_size,(j+1)*cache_size
        t_xcache = Thread(target=load_img_h5,
                          args=(imgs,start,stop,out_xcache))
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


def evaluate_model(model, gen_test, num_test, batch_size):
    Y_pred = np.zeros(num_test)
    Y_true = np.zeros(num_test)
    # ipdb.set_trace()

    n_step = num_test / batch_size

    i = 0
    while i < n_step:
    # for i in range(n_step):
        x, y_true = next(gen_test)
        if x.shape[0] != batch_size:
            continue
        
        y_pred = np.argmax(model.predict_on_batch(x), axis=1)
        y_true = np.argmax(y_true, axis=1)

        Y_true[i*batch_size:(i+1)*batch_size] = y_true
        Y_pred[i*batch_size:(i+1)*batch_size] = y_pred

        i += 1

    print confusion_matrix(Y_true, Y_pred)

    


def main(model_list, train_set, weights_path, test_set = None,
         batch_size=32, batch_per_cache=100, shuffle=False, nb_epoch=10,
         use_class_weight=False, verbose=False):

    f = h5py.File(train_set, "r")
    labels = f["labels"]
    num_train = labels.shape[0]

    n_labels = len(set(labels))
    # Y = np_utils.to_categorical(labels, n_labels)


    count_label = {}
    for l in labels:
        if l not in count_label:
            count_label[l] = 0
        count_label[l] += 1

    if verbose:
        print("Frequency table : ")
        print count_label

    class_weight = {}
    if use_class_weight:
        for k in count_label:
            class_weight[k] = 1./count_label[k]

    
    f.close()
    
    gen_train = ImageGenerator(train_set,224, batch_per_cache=batch_per_cache,
                               batch_size=batch_size, shuffle=shuffle)

    num_test = None
    if test_set != None:
        f = h5py.File(test_set, "r")
        labels = f["labels"]
        num_test = labels.shape[0]
        f.close()
        gen_test = ImageGenerator(test_set,224, batch_per_cache=25,
                                  batch_size=batch_size, shuffle=False)
        


        
    for model_file in model_list:
        print("Training model : "+model_file)
        imp.load_source("convnet", model_file)
        from convnet import model
        model.fit_generator(gen_train,
                            samples_per_epoch=num_train,
                            nb_epoch=nb_epoch,
                            validation_data=gen_test,
                            nb_val_samples=num_test,
                            class_weight=class_weight)

        print("Evaluate train (on an extract) : ")
        evaluate_model(model, gen_train, 1024, batch_size)
        
        print("Evaluate test : ")
        evaluate_model(model, gen_test, num_test, batch_size)
        

        model.save_weights(join(weights_path, "weights_"+ \
                                splitext(basename(model_file))[0] + ".h5"),
                           overwrite=True)

        print "---------------------------------------"
        print "---------------------------------------"

        

        

        
        
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("MODELS", nargs="+",
                        help = ("Model scripts. They should all define a model"
                                "variable"))
    parser.add_argument("DATA_TRAIN", help = "h5 file containing preprocessed train images")
    parser.add_argument("WEIGHTSPATH", help="path to folder for trained weights")

    parser.add_argument("-ts", "--testset", default=None,
                        help = "h5 file containing preprocessed test images")
    parser.add_argument("-bs", "--batchsize", type=int, default=32,
                        help="Batch size. Default : 32")
    parser.add_argument("-bpc", "--batchpercache", type=int, default=100,
                        help="Number of batch in a cache. Default : 100")
    parser.add_argument("-sh", "--shuffle", action="store_true",
                        help="Shuffle the train set while training")
    parser.add_argument("-ep", "--epoch", type=int, default=10,
                        help="Number of epoch")
    parser.add_argument("-wc", "--weightsclass", action="store_true",
                        help=("If this option is given, during the training, "
                              "the weight of the loss will be proportionnal with the number "
                              "of pictures in a class")
                        )
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()

    main(args.MODELS, args.DATA_TRAIN, args.WEIGHTSPATH, test_set=args.testset,
         batch_size=args.batchsize, batch_per_cache=args.batchpercache,
         shuffle=args.shuffle, nb_epoch=args.epoch, use_class_weight=args.weightsclass,
         verbose=args.verbose)
