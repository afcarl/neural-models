#!/usr/bin/env python2

import numpy as np
import argparse

from os.path import isfile, join, splitext, basename
import h5py

from sklearn.cross_validation import StratifiedShuffleSplit
from progressbar import Percentage, Bar, ETA, AdaptiveETA, ProgressBar

from convnetskeras.convnets import preprocess_image_batch

def folder2labels(path):
    #TODO
    pass


def split_traintest(data, splitsize=.1):
    sss = StratifiedShuffleSplit([l for f,l in data], test_size=splitsize)
    for train_index, test_index in sss:
        data_train = [data[i] for i in train_index]
        data_test  = [data[i] for i in  test_index]
    return data_train, data_test





def preprocessing(data, train_set_h5, test_set_h5=None, splitsize=.1):
    if test_set_h5 == None:
        data_train = data
    else:
        data_train, data_test = split_traintest(data, splitsize=splitsize)

    print("Preprocessing train set ...")
    save_img_h5(data_train, train_set_h5, img_size=(256,256))

    if test_set_h5 != None:
        print("Preprocessing test set ...")
        save_img_h5(data_test, test_set_h5, img_size=(256,256))



def save_img_h5(data, h5_file, cache_size=30, img_size=None):
    f = h5py.File(h5_file, "w")
    files = [name for (name,l) in data]
    labels = np.array([l for (name,l) in data])
    n_samples = len(data)

    files_dset = f.create_dataset("files", data=files)
    
    lab_dset = f.create_dataset("labels", data=labels,compression="gzip")
    
    img_dset = f.create_dataset("imgs", (n_samples,3,256,256),
                                dtype='float32',
                                compression="gzip")
    n_step = len(data)/cache_size
    if n_step % cache_size != 0:
        n_step += 1

        
    ############# Progressbar ################
    widgets = [Percentage(),
               ' ', Bar(),
               ' ', ETA(),
               ' ', AdaptiveETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(data))
    pbar.start()
    #########################################
    
    for j in range(n_step):
        X = preprocess_image_batch(files[j*cache_size:(j+1)*cache_size],
                                   img_size=img_size)
        img_dset[j*cache_size:(j+1)*cache_size] = X
        pbar.update(j*cache_size)
        
    pbar.finish()
    f.close()



def main(data_file, train_set, test_set=None, splitsize=.1):
    data = []
    with open(data_file, "r") as f:
        for l in f:
            path, label = l.split(",")
            label = int(label[:-1])
            data.append((path, label))

    preprocessing(data, train_set, test_set_h5 = test_set, splitsize=splitsize)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("DATA", help = "CSV file containing images and labels")
    parser.add_argument("H5PATH", help= "Path to the preprocessed data")
    parser.add_argument("-ts", "--testset", default=None,
                        help = ("If path is given, it splits the data into "
                                "train and test set")
                        )
    parser.add_argument("-ss", "--splitsize", type=float, default=.1,
                        help="Size of the test set. Default : 0.1")
    args = parser.parse_args()


    
    main(args.DATA, args.H5PATH, test_set=args.testset, splitsize=args.splitsize)
    
