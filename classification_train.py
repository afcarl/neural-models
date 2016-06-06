import numpy as np

from os.path import isfile, join, splitext, basename

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





def split_traintest(data):
    sss = StratifiedShuffleSplit([l for f,l in data])
    for train_index, test_index in sss:
        data_train = [data[i] for i in train_index]
        data_test  = [data[i] for i in  test_index]
    return data_train, data_test





        
    

def load_img_h5(h5dataset,start,stop,out=None):
    data = h5dataset[start:stop]
    if not out is None:
        out.append(data)        
    else:
        return data




        
    
def ImageGenerator(data, img_size, batch_per_cache=100, batch_size=16,
                   shuffle=False, **kwargs):
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

    cache_size = batch_size*batch_per_cache

    f = h5py.File(data, "r")
    imgs,labels = f["imgs"],f["labels"]
    n_samples = labels.shape[0]
   
    n_labels = len(set(labels))
    Y = np_utils.to_categorical(labels, n_labels)
    
    n_step = len(data)/cache_size
    if len(data) % cache_size != 0:
        n_step += 1

    permutation = np.random.permutation(n_step) if shuffle else np.arange(n_step)
    k = 0
    j = permutation[k]

    start,stop = j*cache_size,(j+1)*cache_size
    X_cache = load_img_h5(imgs,start,stop)
    Y_cache = Y[start:stop]

    
    datagen.fit(X_cache)

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




def main(model_list, data, weights_path, test_set = None,
         batch_size=32, batch_per_cache=100, shuffle=False, nb_epoch=10):

    f = h5py.File(data, "r")
    labels = f["labels"]
    num_train = labels.shape[0]
    f.close()
    
    gen_train = ImageGenerator(train_set,227, batch_per_cache=batch_per_cache,
                               batch_size=batch_size, shuffle=shuffle)

    if test_set != None:
        f = h5py.File(test_set, "r")
        labels = f["labels"]
        num_test = labels.shape[0]
        f.close()
        gen_test = ImageGenerator(test_set,227, batch_per_cache=25,
                                  batch_size=batch_size, shuffle=False)
    else:
        num_test = None


        
    for model_file in model_list:
        print("Training model : "+model_file)
        imp.load_source("convnet", model_file)
        from convnet import model
        model.fit_generator(gen_train,
                            samples_per_epoch=num_train,
                            nb_epoch=nb_epoch,
                            validation_data=gen_test,
                            nb_val_samples=num_test)

        model.save_weights(join(weights_path, "weights_"+ \
                                splitext(basename(model_file))[0] + ".h5"))

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
                        help="Number of epoch"
    
    args = parser.parse_args()

    main(args.MODELS, args.DATA, args.WEIGHTSPATH, test_set = args.testset,
         batch_size=args.batch_size, batch_per_cache=args.batchpercache,
         shuffle=args.shuffle, n_epoch=args.epoch)
