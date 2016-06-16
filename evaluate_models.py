#!/usr/bin/env python2

import os
from os.path import join, splitext, basename
import imp
import argparse
import h5py


from convnetskeras.convnets import preprocess_image_batch

import ipdb


#import numpy as np
import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from classification_train import evaluate_model, ImageGenerator

def plot_roc_curve(y_test_list, y_pred_list, output_path):

    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i, (y_test, y_pred) in enumerate(zip(y_test_list, y_pred_list)):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
        roc_auc[i] = auc(fpr[i], tpr[i])

    


    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()

    for i in fpr:
        plt.plot(fpr[i], tpr[i], label='ROC curve %i (area = %0.2f)' % (i, roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(output_path)

    


def main(model_list, test_set, weights_path, outputFile,
         batch_size=32, batch_per_cache=None):

    f = h5py.File(test_set, "r")
    labels = f["labels"]
    num_test = labels.shape[0]
    
    f.close()

    gen_test = ImageGenerator(test_set,224, batch_per_cache=batch_per_cache,
                              batch_size=batch_size, shuffle=False,
                              dataaugmentation=False)


    y_test_list = []
    y_pred_list = []
    for model_file in model_list:
        print("Testing model : "+model_file)
        imp.load_source("convnet", model_file)
        from convnet import model

        model.load_weights(join(weights_path,
                                "weights_"+splitext(basename(model_file))[0] + ".h5"))


        
        _, y_test, y_pred = evaluate_model(model, gen_test, num_test, batch_size)
        y_test_list.append(y_test)
        y_pred_list.append(y_pred)
    ipdb.set_trace()
    plot_roc_curve(y_test_list, y_pred_list, "ROC.png")
    

    

            
            
            
    


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("MODELS", nargs="+",
                        help = ("Model script. They should all define a model"
                                "variable"))
    parser.add_argument("DATATEST", help = "h5 file containing test set")
    parser.add_argument("WEIGHTSPATH", help="path to folder for trained weights")
    parser.add_argument("OUTPUT", help="Output directory")

    
    parser.add_argument("-bs", "--batchsize", type=int, default=32,
                        help="Batch size. Default : 32")
    parser.add_argument("-bpc", "--batchpercache", type=int, default=100,
                        help="Number of batch in a cache. Default : 100")
    
    args = parser.parse_args()

    main(args.MODELS, args.DATATEST, args.WEIGHTSPATH, args.OUTPUT,
         batch_size=args.batchsize, batch_per_cache=args.batchpercache)
