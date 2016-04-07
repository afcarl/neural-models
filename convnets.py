import sys

sys.path.insert(0, "/home/lblier/.local/lib/python2.7/site-packages")
sys.path.append("/home/lblier/")

import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Permute, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from customlayers import Convolution2DGroup, CrossChannelNormalization
import numpy as np

from copy import deepcopy


from os.path import join

from scipy.misc import imread, imresize, imsave

import pdb

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

        while model.layers[-1].name != output_layer:
            model.layers.pop()

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
    model = Sequential()

    model.add(ZeroPadding2D((0,0),input_shape=(3,227,227)))
    model.add(Convolution2D(96, 11, 11,
                            subsample=(4,4),
                            activation='relu',
                            name='conv_1'))
    model.add(MaxPooling2D((3, 3), strides=(2,2)))
    model.add(CrossChannelNormalization())

    
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2DGroup(2,256,5,5,
                                 input_shape=model.output_shape,
                                 subsample=(1,1),
                                 activation='relu',
                                 name='conv_2'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(CrossChannelNormalization())

    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(384,3,3,
                            subsample=(1,1),
                            activation='relu',
                            name='conv_3'))


    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2DGroup(2,384,3,3,
                                 input_shape=model.output_shape,
                                 subsample=(1,1),
                                 activation='relu',
                                 name='conv_4'))


    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2DGroup(2,256,3,3,
                                 input_shape=model.output_shape,
                                 subsample=(1,1),
                                 activation='relu',
                                 name='conv_5'))
    model.add(MaxPooling2D((3, 3), strides=(2,2)))

   
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))


    
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))


    
    model.add(Dense(1000, activation='softmax', name='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


    
              

def preprocess_image_batch(image_paths, img_width=224, img_height=224):
    img_list = []
    for im_path in image_paths:
        
        img = imresize(imread(im_path, mode='RGB'), (img_width, img_height))
        img = img.astype('float32')
        # We permute the colors to get them in the BGR order
        #pdb.set_trace()
        img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        # We normalize the colors with the empirical means on the training set
        #img[:, :, 0] -= 103.939
        #img[:, :, 1] -= 116.779
        #img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))
        img_list.append(img)

    img_batch = np.stack(img_list, axis=0)
    return img_batch

def preprocess_image_batch2(image_paths, out=None):
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
    im = preprocess_image_batch2(['cat.jpg'])

    # Test pretrained model
    #model = convnet('alexnet', weights_path='weights/alexnet_weights.h5',convolutionize=False)
    model = load_coeff()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    
