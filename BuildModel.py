from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys


epoch = 10
learning_rate = 0.0004
batch_size = 16
optimizer ='RMSprop'
initial_weights = 'Xavier'

default_values = dict(epoch=10,\
                      learning_rate=0.01,\
                      batch_size=10,\
                      optimizer=Adam,\
                      initial_weights=0,\
                      cnn_class=0,\
                      pre_weights='Xavier',\
                      lstm_conf=(LSTM,dict(units = 256)
                      ))


def build(epoch = default_values["epoch"],\
          learning_rate = default_values["learning_rate"], \
          batch_size = default_values["batch_size"],\
          optimizer = default_values["optimizer"],\
          initial_weights = default_values["initial_weights"],\
          cnn_class = default_values["cnn_class"],\
          pre_weights = default_values["pre_weights"], \
          lstm_conf = default_values["lstm_conf"]):

    model=0






    return model