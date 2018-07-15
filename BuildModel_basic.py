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

def build(size, seq_len , learning_rate ,
          optimizer_class ,\
          initial_weights ,\
          cnn_class ,\
          pre_weights , \
          lstm_conf , \
          cnn_train_type):
    #Create CNN

    input_layer = Input(shape=(seq_len, size, size, 3))



    if(cnn_train_type!='train'):
        cnn = cnn_class(weights=pre_weights,include_top=False)
    else:
        cnn = cnn_class(include_top=False)

    # input_layer2 = Input(shape=(size, size, 3))
    # cnn2 = cnn(input_layer2)
    # model = Model(inputs=input_layer2, outputs=cnn2)
    # print model.summary()

    #control Train_able of CNNN
    if(cnn_train_type=='static'):
        for layer in cnn.layers:
            layer.trainable = False
    if(cnn_train_type=='retrain'):
        for layer in cnn.layers:
            layer.trainable = True

    cnn = TimeDistributed(cnn)(input_layer)

    # model = Model(inputs=input_layer, outputs=cnn)
    # print model.summary()

    # lstm= ConvLSTM2D(filters=256, kernel_size=(3, 3),padding='same', return_sequences=False)(cnn)
    lstm = lstm_conf[0](**lstm_conf[1])(cnn)
    flat = Flatten()(lstm)
    dense = BatchNormalization()(flat)
    #dense = Dense(1000,activation= 'relu', kernel_initializer =initial_weights)(dense)
    dense = Dense(256, activation= 'relu', kernel_initializer=initial_weights)(dense)
    dense = Dense(10, activation= 'relu', kernel_initializer=initial_weights)(dense)
    predictions = Dense(1,  activation='sigmoid')(dense)

    model = Model(inputs=input_layer, outputs=predictions)
    optimizer = optimizer_class[0](lr=learning_rate, **optimizer_class[1])
    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['acc'])

    print(model.summary())

    return model