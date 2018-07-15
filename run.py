from keras.optimizers import RMSprop

import BuildModel
import pandas as pd
from keras.applications import Xception, ResNet50, InceptionV3
from keras.layers import LSTM, ConvLSTM2D
from sklearn.metrics import accuracy_score

import BuildModel_basic
import DatasetBuilder

datasets_videos = dict(hocky ="data/raw_videos/HockeyFights")
datasets_frames = "data/raw_frames"
figure_size = 244
split_ratio = 0.8
epochs = 10
learning_rate = 0.0004
batch_size = 16
load_all = True
train_path, test_path,train_y, test_y, avg_length = DatasetBuilder.createDataset(datasets_videos, datasets_frames, figure_size, split_ratio)
if load_all:
    train_x, train_y = DatasetBuilder.load_data(train_path,train_y,figure_size,avg_length)
else:
    train_gen = DatasetBuilder.data_generator(train_path,train_y,batch_size,figure_size,avg_length)

test_x, test_y = DatasetBuilder.load_data(test_path,test_y,figure_size,avg_length)

optimizer =(RMSprop,{})
initial_weights = 'Xavier'

weights='imagenet'
cnn_train_types = ['retrain', 'static', 'train']
cnns_pretrained = dict(Xception = Xception, ResNet50 = ResNet50, InceptionV3 =InceptionV3)

lstms_type = dict(
    LSTM = (LSTM,dict(units = 256)),
    COV_LSTM = (ConvLSTM2D, dict(filters=256, kernel_size=3))
                  )

results = []
for cnn_train_type in cnn_train_types:
    for cnn_name, cnn_class in cnns_pretrained.iteritems():
        for lstm_name, lstm_conf in lstms_type.iteritems():
            result = dict(cnn_train = cnn_train_type,cnn = cnn_name, lstm = lstm_name,epochs = epochs,learning_rate = learning_rate, batch_size = batch_size,
                                     optimizer = RMSprop.__class__.__name__, initial_weights = initial_weights,seq_len = avg_length)

            model = BuildModel_basic.build(seq_len = avg_length, learning_rate = learning_rate,
                                           optimizer_class = optimizer, initial_weights = initial_weights,
                                     cnn_class = cnn_class,pre_weights = weights, lstm_conf = lstm_conf)
            if load_all:
                model.fit(train_x, train_y, epochs=epochs,batch_size = batch_size)
            else:
                model.fit_generator(
                    generator=train_gen,
                    epochs=epochs,
                    batch_size = batch_size,
                    verbose=1,
                    validation_steps=40,
                    workers=4)
            preds = model.predcit(test_x)
            result['accuracy'] = accuracy_score(preds, test_y)
            results.append(result)
            print result
pd.DataFrame(results).to_csv("results.csv")