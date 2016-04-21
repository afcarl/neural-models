import sys

sys.path.insert(0, "/home/lblier/.local/lib/python2.7/site-packages")
sys.path.append("/home/lblier/")

import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from customlayers import convolution2Dgroup, crosschannelnormalization, splittensor
import numpy as np

from copy import deepcopy
#from joblib import Parallel, delayed
from multiprocessing import Pool

from os.path import join

from scipy.misc import imread, imresize, imsave

import pdb

from time import time

def convnet(network, weights_path=None, output_layer=None, convolutionize=False,
            trainable=None):
    """
    Returns a keras model for a CNN.
    
    This model takes as an input a 3x224x244 picture and returns a 
    1000-dimensional vector.

    It can also be used to look at the hidden layers of the model.

    It can be used that way : 
    >>> im = preprocess_img_batch(['cat.jpg'])

    >>> # Test pretrained model
    >>> model = convnet('vgg_16', 'vgg16_weights.h5', 'conv5_3')
    >>> sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    >>> model.compile(optimizer=sgd, loss='categorical_crossentropy')
    >>> out = model.predict(im)

    Parameters
    --------------
    network: str
        The type of network chosen. For the moment, can be 'vgg_16' or 'vgg_19'

    weights_path: str
        Location of the pre-trained model. If not given, the model will be trained

    layer_output: dict
        Iterable of the name of all the feature layers that will be returned

    convolutionize: bool
        Says wether the fully connected layers are transformed into Convolution2D layers


    Returns
    ---------------
    model:
        The keras model for this convnet

    output_dict:
        Dict of feature layers, asked for in output_layers.
    """

    
    # Select the network
    if network == 'vgg_16':
        model = VGG_16(weights_path)
        
    elif network == 'vgg_19':
        model = VGG_19(weights_path)

    elif network == 'alexnet':
        model=AlexNet(weights_path)
    else:
        raise ValueError("Network "+network+" is not known")

    

    # Select the output
    if output_layer != None:
        # Check that the layer name exists
        if not any(layer.name == output_layer for layer in model.layers):
            raise ValueError("Layer "+output_layer+" does not exist in this network")

        layer = next(layer for layer in model.layers if layer.name == output_layer)
        model = Model(input=model.input,output=layer.output)

    if trainable != None:
        for layer in model.layers:
            if layer.name in trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    if convolutionize:
        conv_stride = (5, 5)
        mod_conv = Sequential()

        first_dense = True
        for layer in model.layers:
            layer_type = layer.get_config()['name']
            if  layer_type == "Dense":
                n_previous_filters = mod_conv.output_shape[1]
                W, b = layer.get_weights()
                new_size = int(np.sqrt(W.shape[0]/n_previous_filters))
                new_W = W.reshape((W.shape[1],
                                   n_previous_filters,
                                   new_size,
                                   new_size))
                if first_dense:
                    subsample = conv_stride
                else:
                    subsample = (1, 1)
                new_layer = Convolution2D(W.shape[1], new_size, new_size,
                                          weights=[new_W, b],
                                          subsample=subsample,
                                          activation=layer.get_config()["activation"])
                mod_conv.add(new_layer)
                
                
            elif layer_type == "Flatten":
                pass
            else:
                mod_conv.add(layer)
            
            if len(mod_conv.layers) == 1:
                mod_conv.layers[0].set_input_shape((None, 3, None, None))

        return mod_conv
    return model

    
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224))) 
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, name='softmax', activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
    return model


def VGG_19(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, name='softmax', activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def AlexNet(weights_path=None):
    inputs = Input(shape=(3,227,227))
    
    conv_1 = ZeroPadding2D((0,0),input_shape=(3,227,227))(inputs)
    conv_1 = Convolution2D(96, 11, 11,
                           subsample=(4,4),
                           activation='relu',
                           name='conv_1')(conv_1)
    
    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization()(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    
    conv_2_1 = splittensor(ratio_split=2,id_split=0)(conv_2)
    conv_2_1 = Convolution2D(128,5,5,subsample=(1,1),
                             activation="relu",
                             name='conv_2_1')(conv_2_1)
    
    conv_2_2 = splittensor(ratio_split=2,id_split=1)(conv_2)
    conv_2_2 = Convolution2D(128,5,5,subsample=(1,1),
                             activation="relu",
                             name='conv_2_2')(conv_2_2)

    conv_2 = merge([conv_2_1,conv_2_2], mode='concat',concat_axis=1)
    
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,
                           subsample=(1,1),
                           activation='relu',
                           name='conv_3')(conv_3)

    
    conv_4 = ZeroPadding2D((1,1))(conv_3)

    conv_4_1 = splittensor(ratio_split=2,id_split=0)(conv_4)
    conv_4_1 = Convolution2D(192,3,3,subsample=(1,1),
                             activation="relu",
                             name='conv_4_1')(conv_4_1)
    
    conv_4_2 = splittensor(ratio_split=2,id_split=1)(conv_4)
    conv_4_2 = Convolution2D(192,3,3,subsample=(1,1),
                             activation="relu",
                             name='conv_4_2')(conv_4_2)

    conv_4 = merge([conv_4_1,conv_4_2], mode='concat',concat_axis=1)
    
    conv_5 = ZeroPadding2D((1,1))(conv_4)

    conv_5_1 = splittensor(ratio_split=2,id_split=0)(conv_5)
    conv_5_1 = Convolution2D(128,3,3,subsample=(1,1),
                             activation="relu",
                             name='conv_5_1')(conv_5_1)
    
    conv_5_2 = splittensor(ratio_split=2,id_split=1)(conv_5)
    conv_5_2 = Convolution2D(128,3,3,subsample=(1,1),
                             activation="relu",
                             name='conv_5_2')(conv_5_2)

    conv_5 = merge([conv_5_1,conv_5_2], mode='concat',concat_axis=1)

    
    dense_1 = MaxPooling2D((3, 3), strides=(2,2))(conv_5)
    dense_1 = Flatten()(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)

    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)

    prediction = Dropout(0.5)(dense_2)  
    prediction = Dense(1000, activation='softmax',name='softmax')(prediction)

    model = Model(input=inputs, output=prediction)
    
    if weights_path:
        model.load_weights(weights_path)

    return model



def load_coeff(path='parameters_releasing/'):
    model = convnet('alexnet')
    suf = '_65.npy'
    W_list = []
    b_list = []
    for i in range(8):
        if i in [1, 3, 4]:
            W0, W1 = np.load(path+'W0_'+str(i)+suf), np.load(path+'W1_'+str(i)+suf)
            b0, b1 = np.load(path+'b0_'+str(i)+suf), np.load(path+'b1_'+str(i)+suf)

            W0 = W0.transpose((3, 0, 1, 2))
            W1 = W1.transpose((3, 0, 1, 2))
            W_list.append([W0, W1])
            b_list.append([b0, b1])
        else:
            W = np.load(path+'W_'+str(i)+suf)
            b = np.load(path+'b_'+str(i)+suf)
            if i in [0, 2]:
                W = W.transpose((3, 0, 1, 2))
            W_list.append(W)
            b_list.append(b)


    

    for i in [1,3]:
        layer = next(layer for layer in model.layers if layer.name == 'conv_'+str(i))
        layer.set_weights([W_list[i-1], b_list[i-1]])

    for i in [2,4,5]:
        for j in [1,2]:
            layer = next(layer for layer in model.layers \
                         if layer.name == '_'.join(['conv',str(i),str(j)]))
            layer.set_weights([W_list[i-1][j-1], b_list[i-1][j-1]])


    for i in [1,2]:
        layer = next(layer for layer in model.layers if layer.name == 'dense_'+str(i))
        layer.set_weights([W_list[i+4], b_list[i+4]])

    layer = next(layer for layer in model.layers if layer.name == 'softmax')
    layer.set_weights([W_list[7], b_list[7]])

    return model
    
              

def preprocess_image_batch(image_paths, img_width=256, img_height=256, out=None):
    img_list = []

    img_mean = np.load("../NeuralModels/img_mean.npy")
    img_mean = img_mean.astype('float32')

    
    for im_path in image_paths:
        img = imresize(imread(im_path, mode='RGB'), (img_width, img_height))
        img = img.astype('float32')
        # We permute the colors to get them in the BGR order
        #pdb.set_trace()
        #img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        # We normalize the colors with the empirical means on the training set
        # img[:, :, 0] -= 103.939
        # img[:, :, 1] -= 116.779
        # img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))
        img -= img_mean
        img_list.append(img)

    img_batch = np.stack(img_list, axis=0)
    if not out is None:
        out.append(img_batch)
    else:
        return img_batch





    
def preprocess_image_batch2(image_paths, out=None, n_jobs=1):
    img_list = []
    img_size = 256
    crop_size = 227

    img_mean = np.load("img_mean.npy")
    img_mean = img_mean.astype('float32')
    
    for im_path in image_paths:
        img = imresize(imread(im_path, mode='RGB'), (img_size, img_size))
        img = img.astype('float32')
        # We permute the colors to get them in the BGR order
        #pdb.set_trace()
        img = img.transpose((2, 0, 1))
        img = img - img_mean
        img = img[:, (img_size-crop_size)/2:-(img_size-crop_size)/2,
                  (img_size-crop_size)/2:-(img_size-crop_size)/2]
        img_list.append(img)

    
    p = Pool(processes=n_jobs)
    img_list = p.map(Preprocessor(img_size,img_mean,crop_size),
                     image_paths)

    img_batch = np.stack(img_list, axis=0)
    if out == None:
        return img_batch
    else:
        out.append(img_batch)

def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x




if __name__ == "__main__":

    # base_image = K.variable(preprocess_image('~/Pictures/cat.jpg'))
    #im = preprocess_image_batch2(['cat.jpg'])

    # Test pretrained model
    #model = convnet('alexnet',convolutionize=False)
    model = load_coeff()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    
