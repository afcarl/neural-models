#!/usr/bin/env python2

import os
import imp
import argparse

from convnetskeras.convnets import preprocess_image_batch

import ipdb




    


def main(model_file, img_list, weights_path, batch_size=32, batch_per_cache=None):
    imp.load_source("convnet", model_file)
    from convnet import model
    model.load_weights(weights_path)

    data = []
    with open(img_list, "r") as f:
        for l in f:
            data.append(l[:-1])

    print("File paths loaded")
    output = []
    i = 0
    #while i < len(data):
    f = open("img_classify.csv", "w")
    for i in range(0, len(data), batch_size):
        print i
        try:
            X = preprocess_image_batch(data[i:i+batch_size],
                                       img_size=(224,224))
        except:
            print("BUG")
            continue
            
        Y_pred = model.predict_on_batch(X)
        output.extend(zip(data[i:i+batch_size], list(Y_pred)))
        for j, path_img in enumerate(data[i:i+batch_size]):
            #ipdb.set_trace()
            f.write(os.path.basename(path_img)+";")
            f.write(";".join((str(l)+";"+str(s) for (l,s) in enumerate(list(Y_pred[j])))))
            f.write("\n")

    f.close()
            
            
            
    


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("MODELS",
                        help = ("Model script. They should all define a model"
                                "variable"))
    parser.add_argument("DATA", help = "jpg files to classify")
    parser.add_argument("WEIGHTSPATH", help="path to folder for trained weights")

    
    parser.add_argument("-bs", "--batchsize", type=int, default=32,
                        help="Batch size. Default : 32")
    parser.add_argument("-bpc", "--batchpercache", type=int, default=100,
                        help="Number of batch in a cache. Default : 100")
    
    args = parser.parse_args()

    main(args.MODELS, args.DATA, args.WEIGHTSPATH,
         batch_size=args.batchsize, batch_per_cache=args.batchpercache)
