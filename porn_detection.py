#!/usr/bin/python
# Usage:
#   $ python porn_detection.py 0.0.0.0 5234 /path/to/porn.h5 &


from flask import Flask, request
import json
import hashlib
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import logging
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pickle as pkl
import random
import sys
import tempfile
import traceback
from werkzeug import secure_filename
from werkzeug.contrib.cache import SimpleCache

from convnets import convnet, preprocess_image_batch, preprocess_image_batch2

app = Flask(__name__)

# Set logging level to error
log = logging.getLogger('werkzeug')
loghandler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(loghandler)
log.setLevel(logging.ERROR)

# Create a cache, in case an image is resubmitted
CACHE_TIMEOUT = 300
cache = SimpleCache()

@app.route("/detect", methods=['POST'])
def detect():
    global model
    try:
      fileids = []
      tempfiles = []
      filehashes = []
      cachedresults = []
      for (fileid, file) in request.files.items():
        hash = hashlib.sha256(file.read()).digest()
        prob = cache.get(hash) if caching else None
        if prob:
          cachedresults.append((fileid, prob))
        else:
          tmp = tempfile.NamedTemporaryFile()
          file.seek(0)
          file.save(tmp.name)
          fileids.append(fileid)
          tempfiles.append(tmp)
          filehashes.append(hash)
      if len(tempfiles) + len(cachedresults) > 0:
        y = []
        if len(tempfiles) > 0:
          img_paths = [tmp.name for tmp in tempfiles]
          X = preprocess_image_batch2(img_paths)
          y = model.predict(X)
          for tmp in tempfiles:
            tmp.close()
        if caching:
          for (hash, prob) in zip(filehashes, y):
            cache.set(hash, prob, CACHE_TIMEOUT)
        probs = [{"id": fileid, "prob": prob[0]} for (fileid, prob) in zip(fileids, y) + cachedresults]
        return json.dumps({ "results": probs })
      else:
        return None
    except:
      print("Exception occured: ")
      traceback.print_exc(file=sys.stdout)
      return "{}", 500

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage: "+sys.argv[0]+" HOST PORT PATH_TO_PORNH5 CACHING"
        sys.exit(1)

    pornH5Path = sys.argv[3]
    if not pornH5Path.endswith('porn.h5'):
        pornH5Path = os.path.join(pornH5Path, 'porn.h5')

    caching = False
    if len(sys.argv) > 4:
        caching = bool(sys.argv[4])

    model = convnet('alexnet', output_layer='dense_2')
    model.add(Dense(1, activation='sigmoid', name='classifier'))
    model.load_weights(pornH5Path)
    
    sgd = SGD(lr=.5, decay=1.e-6, momentum=0., nesterov=False)
    model.compile(optimizer=sgd, loss='binary_crossentropy')

    print "Ready"
    app.run(debug=False, host=sys.argv[1], port=sys.argv[2], threaded=False)
