# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:57:45 2016

@author: hedi
"""

from convnets import preprocess_image_batch as preprocess
from convnets import convnet
from keras.optimizers import SGD
from keras.models import Graph
from keras.layers import Dense
from keras.regularizers import l2
import keras.backend as K
from time import time
import numpy as np

def HierarchicalCNN(dimensions, 
                    convnet_name='alexnet',weights_path=None, 
                    output_layer="dense_2"):
    """
    Hierarchical CNN
    Computes visual features via convnet, and carries these features into a 
    hierarchical classifier where we have n_binary levels of binary 
    classification, followed by a final classification step with n_out classes.
    
    The loss function is a combination between the n_binary+1 losses
    """
    hierarchy = Graph()
    cnn = convnet(convnet_name, weights_path, output_layer, 
                  convolutionize = False, trainable=trainable)
    
    input_shape = cnn.layers[0].input_shape[1:]
    hierarchy.add_input('input_image', input_shape=input_shape)
    hierarchy.add_node(cnn,'feat_0', input='input_image')
    for n,d in enumerate(dimensions):
        if n+1<len(dimensions):
            dim_out = 2
            emb = Dense(d, activation='relu')
            hierarchy.add_node(emb, name='feat_'+str(1+n), input='feat_'+str(n))
        else:
            dim_out = d
        out = Dense(dim_out, activation='softmax', W_regularizer=l2())
        hierarchy.add_node(out, name='softmax'+str(n), input='feat_'+str(n))
        
    hierarchy.add_output(name='output', 
                         inputs=[o for o in hierarchy.nodes if 'softmax' in o], 
                                 merge_mode="concat")
    return hierarchy
    
def hierarchical_loss(y_true, y_pred, weights):
    t = y_true[:,:2]
    p = y_pred[:,:2]
    loss = -weights[0]*K.categorical_crossentropy(t,p)
    for idx in xrange(1, len(weights)-1):
        b = t[:,0]
        w = weights[idx]
        p = y_pred[:, 2*idx:2*(idx+1)]
        t = y_true[:, 2*idx:2*(idx+1)]
        loss -= w*b*K.categorical_crossentropy(t, p)
    idx += 1
    b = t[:,0]
    w = weights[idx]
    t = y_true[:,2*idx:]
    p = y_pred[:, 2*idx:]
#    loss -= w*b*K.categorical_crossentropy(t,p)
    return K.mean(loss)


def labels_to_onehot(l):
    label = l.split()
    y = np.zeros(4 + n_classes)
    if label[0] == 'bag':
        y[0] = 1 #it's a bag
    else:
        y[1] = 1 #it's not a bag
        return y 
    if label[1] == 'UNK':
        return y #it's a bag, and we dont know if it's vuitton
    elif label[1] == 'vuitton':
        y[2] = 1 #it's vuitton
    else:
        y[3] = 1 #it's not vuitton
        return y 
    if label[2] == 'UNK':
        return y #it's a bag, vuitton, but we don't know which one
    else:
        idx = cat_to_idx[label[2]] #retrieve the category index
        y[4 + idx] = 1 # 
        return y
        
if __name__ == "__main__":
    n_classes = 10
    dimensions = [1000, 200, n_classes]
    weights = [1., 0.5, 1./n_classes]
    n_binary = len(dimensions)-1
    convnet_name = 'alexnet'
    weights_path='weights/alexnet_weights.h5' 
    output_layer='dense_2'
    trainable = False
    # Test pretrained model
    
    #%%
    lv_bags = u"steamer twist alma speedy malle lockit \
    hologram cluny dora tournon".split()
    lv_bags = list(set(lv_bags))
    cat_to_idx = dict((k,i) for i,k in enumerate(lv_bags))
    

    
    
    model = HierarchicalCNN(dimensions, 'alexnet',
                            weights_path=weights_path, 
                            output_layer='dense_2')
                            

    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    loss = lambda y_true,y_pred: hierarchical_loss(y_true,y_pred,weights)
    losses = dict((k,loss) for k in model.outputs)
    tic = time()
    model.compile(optimizer=sgd, loss=losses)
    toc = time()-tic
    print "Compilation took %fsec" % (toc)
    
    im = preprocess(['bag.jpg', 'twitterBag/CcUqJ2nXEAATjiD.jpg'], 227, 227)
    labels = ['bag novuitton UNK', 'nobag vuitton UNK']
    y_true = []
    for l in labels:
        y_true.append(labels_to_onehot(l))
    y_true = np.array(y_true, 'int')
    
    out = model.predict({'input_image':im})
    y_pred = out['output']
    print "y_pred"
    print y_pred
    print "y_true"
    print y_true

    
    print model.loss['output'](y_true, y_pred).eval()
    print model.train_on_batch({'input_image':im, 'output':y_true})
