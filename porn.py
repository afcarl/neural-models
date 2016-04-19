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

from classification_train import ImageGenerator, preprocessing

import pickle as pkl

import threading
#from multiprocessing import Pool

import random
import time
import pdb


#########################################
####### WE BUILD THE DATA LIST ##########

mypath='/mnt/data/datasets/flickrdataset/'

labels={'bikinis':0, 'lingeries':1, 'porn':2, 'porn/realgirls':2,
        'swimsuits':3,'sports_filtered':4,'women_filtered':5,
        'babies_filtered':6,'imagenet':7}


folders = ['bikinis','lingeries','porn', 'porn/realgirls', 'swimsuits','sports_filtered','women_filtered','babies_filtered']


files = [(join(mypath,fold,f),labels[fold]) for fold in folders \
         for f in listdir(join(mypath, fold)) if f[-3:].lower() == "jpg"] 

image_folder = "/mnt/data/lblier/ImageNet/"
files.extend([(join(image_folder,"ILSVRC2012_val_"+\
                    str(i+1).zfill(8)+".JPEG"),7) \
              for i in range(10000)])

mypath = "/mnt/data/lblier/"

files.extend([(join(mypath,fold,f),2) for fold in ["ebony","gayporn"] \
              for f in listdir(join(mypath, fold))])


data = files
data = [(f, 1) if l in [0,1,2,3] else (f,0) for (f,l) in data]


##########################################
######### WE PREPROCESS THE DATA #########
preprocessing_done = True
if not preprocessing_done:
    
    print("Preprocessing of the data ...")
    train_files, test_files,train_set, test_set = preprocessing(data, "porn_classif/") 

else:
    train_files = pkl.load(open("porn_classif/train_files.pkl","r"))
    test_files = pkl.load(open("porn_classif/test_files.pkl","r"))
    train_set = "porn_classif/train_set.h5"
    test_set = "porn_classif/test_set.h5"

    

    
###########################################
######### WE CHOOSE THE MODEL #############

random.shuffle(data)

model = convnet('alexnet', weights_path='weights/alexnet_weights.h5',
                output_layer='dense_2',
                trainable=["None"])
model.add(Dense(1,
                activation='sigmoid',
                name='classifier',
                W_regularizer=l2(0.001)))
sgd = SGD(lr=.1, decay=1.e-6, momentum=0.9, nesterov=False)
model.compile(optimizer=sgd, loss='binary_crossentropy')


batch_size = 16
       


gen_train = ImageGenerator(train_set,227,
                           batch_per_cache=100,
                           batch_size=batch_size,
                           shuffle=False)
gen_test = ImageGenerator(test_set,227,
                          batch_per_cache=100,
                          batch_size=batch_size,
                          shuffle=True)

model.fit_generator(gen_train,
                    samples_per_epoch=len(train_files),
                    nb_epoch=10,
                    validation_data=gen_test,
                    nb_val_samples=len(test_files),
                    show_accuracy=True)
 























# for e in range(n_epoch):
#     for i in range(0,len(data),max_n_pic):
        
#         print "Processing images "+str((i, i+batch_size))
#         files_processed = [f for (f,l) in data[i:i+max_n_pic]]
#         X_train = preprocess_image_batch2(files_processed)
#         Y_train = Y[i:i+max_n_pic]

#         if i == 0:
#             datagen.fit(X_train)

#         model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
#                             samples_per_epoch=X_train.shape[0], nb_epoch=5)
    
    
# for i in range(0,len(data), batch_size):
    
    
#     img_batch = preprocess_image_batch2(files_processed)
#     features_batch = feat.predict(img_batch)
#     data_processed.extend(features_batch)

# # Y = [l for (f,l) in data]

# # with open('/mnt/data/datasets/flickrdataset/processed_conv.pkl', 'wb') as f:
# #     pkl.dump((data_processed,Y), f)
    
# with open('/mnt/data/datasets/flickrdataset/processed_conv.pkl', 'rb') as f:
#     X, y = pkl.load(f)



# X = np.stack(X)
# Y = np.array([1 if x in [0, 1, 2, 3] else 0 for x in y])

# np.random.seed(123)

# X_reject, Y_reject = X[Y==0], Y[Y==0]
# X_accept, Y_accept = X[Y==1], Y[Y==1]

# perm_reject = np.random.permutation(X_reject.shape[0])
# perm_accept = np.random.permutation(X_accept.shape[0])
# X_reject, Y_reject = X_reject[perm_reject], Y_reject
# X_accept, Y_accept = X_accept[perm_accept], Y_accept

# X_test, Y_test = np.vstack((X_reject[-1000:],X_accept[-1000:])), \
#                  np.concatenate([Y_reject[-1000:],Y_accept[-1000:]])
# X_train, Y_train = np.vstack((X_reject[:8000],X_accept[:8000])), \
#                    np.concatenate([Y_reject[:8000],Y_accept[:8000]])

# perm_test = np.random.permutation(X_test.shape[0])
# perm_train = np.random.permutation(X_train.shape[0])
# X_test, Y_test = X_test[perm_test], Y_test[perm_test]
# X_train, Y_train = X_train[perm_train], Y_train[perm_train]


# Y_train = np_utils.to_categorical(Y_train, 2)
# Y_test = np_utils.to_categorical(Y_test, 2)



# model = Sequential()

# model.add(Dense(2, activation='softmax', name='classifier'))

# sgd = SGD(lr=.5, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')

# batch_size = 128
# nb_epoch = 1000
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# model.save_weights('porn_weights2.h5')



# # model.load_weights("porn_weights.h5")
# # sgd = SGD(lr=.5, decay=1e-6, momentum=0.9, nesterov=True)
# # model.compile(optimizer=sgd, loss='binary_crossentropy')
# # final_model = Sequential()
# # final_model.add(feat)
# # final_model.add(model)




# ##########
# ## Test Imagenet
# ##########
# # image_folder = "/mnt/data/lblier/ImageNet/"
# # files = [join(image_folder,"ILSVRC2012_val_"+\
# #               str(i+1).zfill(8)+".JPEG") \
# #          for i in range(10000)]

# # feat = convnet('alexnet', weights_path='weights/alexnet_weights.h5', output_layer='dense_2')


# # data_processed = []
# # sgd2 = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# # feat.compile(optimizer=sgd2, loss='binary_crossentropy')
# # batch_size = 16
# # score = 0
# # for i in range(0,len(files), batch_size):
# #     print "Processing images "+str((i, i+batch_size))
# #     files_processed = files[i:i+batch_size]
# #     img_batch = preprocess_image_batch2(files_processed)
# #     features_batch = feat.predict(img_batch)
# #     # out = final_model.predict(img_batch)
# #     # score += float((out > 1./2).sum())
# #     # print score/ ((i+1)*batch_size)
# #     data_processed.extend(features_batch)
