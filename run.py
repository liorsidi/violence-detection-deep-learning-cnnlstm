from keras.optimizers import RMSprop

import BuildModel
import pandas as pd
from keras.applications import Xception, ResNet50, InceptionV3, MobileNet
from keras.layers import LSTM, ConvLSTM2D
from sklearn.metrics import accuracy_score

import BuildModel_basic
import DatasetBuilder

datasets_videos = dict(hocky ="data/raw_videos/HockeyFights")
datasets_frames = "data/raw_frames"
figure_size = 244
split_ratio = 0.8
epochs = 500
learning_rate = 0.0004
batch_size = 16
generator_batch_size = 4
force = False

train_path, test_path,train_y, test_y, avg_length = DatasetBuilder.createDataset(datasets_videos, datasets_frames, force)

# load_all = False
# if load_all:
#     train_x, train_y = DatasetBuilder.load_data(train_path,train_y,figure_size,avg_length)
#     test_x, test_y = DatasetBuilder.load_data(test_path, test_y, figure_size, avg_length)
# else:
train_gen = DatasetBuilder.data_generator(train_path,train_y,generator_batch_size,figure_size,avg_length)
test_gen = DatasetBuilder.data_generator(test_path,test_y,generator_batch_size,figure_size,avg_length)
test_x, test_y = DatasetBuilder.load_data(test_path, test_y, figure_size, avg_length)

optimizer =(RMSprop,{})
initial_weights = 'glorot_uniform'

weights='imagenet'
cnn_train_types = ['static']#'static', 'train','retrain',]
cnns_pretrained = dict(Xception = Xception, ResNet50 = ResNet50, InceptionV3 =InceptionV3)

lstms_type = dict(
  #  LSTM = (LSTM,dict(units = 256)),
    COV_LSTM = (ConvLSTM2D, dict(filters=256, kernel_size=(3, 3),padding='same', return_sequences=False))
                  )

results = []
for cnn_train_type in cnn_train_types:
    for cnn_name, cnn_class in cnns_pretrained.items():
        for lstm_name, lstm_conf in lstms_type.items():
            result = dict(cnn_train = cnn_train_type,cnn = cnn_name, lstm = lstm_name,epochs = epochs,learning_rate = learning_rate, batch_size = batch_size,
                                     optimizer = RMSprop.__class__.__name__, initial_weights = initial_weights,seq_len = avg_length)

            model = BuildModel_basic.build(size = figure_size, seq_len = avg_length, learning_rate = learning_rate,
                                           optimizer_class = optimizer, initial_weights = initial_weights,
                                     cnn_class = cnn_class,pre_weights = weights, lstm_conf = lstm_conf,cnn_train_type = cnn_train_type)
            # if load_all:
            #     model.fit(train_x, train_y, epochs=epochs,batch_size = batch_size)
            # else:
            model.fit_generator(
                steps_per_epoch = int(batch_size/generator_batch_size),
                generator=train_gen,
                epochs=epochs,
                )
            evals = model.evaluate(test_x,test_y,batch_size=generator_batch_size)
            result['loss'] = evals[0]
            result['accuracy'] = evals[1]
            results.append(result)
            print(result)
pd.DataFrame(results).to_csv("results.csv")