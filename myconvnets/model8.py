from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation

from convnetskeras. convnets import convnet

trainable_layers = []




vgg16 = convnet('vgg_16',
                  weights_path='/srv/data/convnetsweights/vgg16_weights.h5',
                  trainable=["None"])

for l in vgg16.layers:
    if l.name in trainable_layers:
        pass
    else:
        l.trainable = False
        

input = vgg16.input
img_representation = vgg16.get_layer("flatten").output

classifier = Dense(2,name='classifier',
                   W_regularizer=l2(0.001))(img_representation)
classifier = Activation("softmax", name="softmax")(classifier)

model = Model(input=input,output=classifier)

sgd = SGD(lr=.1, decay=1.e-6, momentum=0.9, nesterov=False)



model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=["accuracy"])
