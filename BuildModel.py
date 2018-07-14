from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys
import logging
from keras.applications import Xception, ResNet50, InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


logger = logging.getLogger('Builder_moudle')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


epoch = 10
learning_rate = 0.0004
batch_size = 16
optimizer ='RMSprop'
initial_weights = 'Xavier'

default_values = dict(epoch=10,\
                      learning_rate=0.0004,\
                      batch_size=16,\
                      optimizer=Adam,\
                      initial_weights=0,\
                      cnn_class=Xception,\
                      pre_weights='Xavier',\
                      lstm_conf=(LSTM,dict(units = 256)),\
                      cnn_train_type='static'
                      )


def build(epoch = default_values["epoch"],\
          learning_rate = default_values["learning_rate"], \
          batch_size = default_values["batch_size"],\
          optimizer = default_values["optimizer"],\
          initial_weights = default_values["initial_weights"],\
          cnn_class = default_values["cnn_class"],\
          pre_weights = default_values["pre_weights"], \
          lstm_conf = default_values["lstm_conf"], \
          cnn_train_type=default_values["cnn_train_type"]):

    model=0
    #Create CNN
    if(cnn_train_type!='train'):
        logger.info("CNN Created with Pre-weights:{}".format(pre_weights))
        base_model = cnn_class(weights=pre_weights,include_top=False)
    else:
        logger.info("CNN Created with no Pre-weights")
        base_model = cnn_class()

    #control Train_able of CNNN
    if(cnn_train_type=='static'):
        logger.info("CNN set to NOT-Train")
        for layer in base_model.layers:
            layer.trainable = False
    if(cnn_train_type=='retrain'):
        logger.info("CNN set to retrain")
        for layer in base_model.layers:
            layer.trainable = True

    # print(base_model.summary())
    # add a global spatial average pooling layer
    x = base_model.output
    logger.info("base_model.output: {}".format(base_model.output))
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(100 , activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    model.summary()

    return model