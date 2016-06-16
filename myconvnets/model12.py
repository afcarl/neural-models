from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation

from convnetskeras. convnets import convnet

trainable_layers = ["dense_1", "conv5_1", "conv5_2",  "conv5_3",
                    "conv4_1", "conv4_2",  "conv4_3",
                    "conv3_1", "conv3_2",  "conv3_3",
                    "conv2_1", "conv2_2"]





vgg16 = convnet('vgg_16',
                  weights_path='/srv/data/convnetsweights/vgg16_weights.h5',
                  trainable=["None"])

for l in vgg16.layers:
    if l.name in trainable_layers:
        pass
    else:
        l.trainable = False
        

input = vgg16.input
img_representation = vgg16.get_layer("dense_1").output

classifier = Dense(2,name='classifier',
                   W_regularizer=l2(0.01))(img_representation)
classifier = Activation("softmax", name="softmax")(classifier)

model = Model(input=input,output=classifier)

opt = SGD(lr=.1, decay=1.e-6, momentum=0.9, nesterov=False)
#opt = Adam()



model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=["accuracy"])
