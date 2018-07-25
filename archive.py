import os
from keras.optimizers import RMSprop, Adam

import BuildModel
import pandas as pd
from keras.applications import Xception, ResNet50, InceptionV3, MobileNet, VGG19
from keras.layers import LSTM, ConvLSTM2D
from sklearn.metrics import accuracy_score

import BuildModel_basic
import DatasetBuilder

datasets_videos = dict(only_hocky = dict(hocky ="data/raw_videos/HockeyFights"))
datasets_frames = "data/raw_frames"
res_path = "results"
figure_size = 244
split_ratio = 0.8
epochs = 100
batch_size = 4
fix_len = 20
initial_weights = 'glorot_uniform'
weights='imagenet'
force = False


optimizer =(RMSprop,dict(decay=0.5)) #Adam, {},
learning_rates = [1e-4]
cnn_train_types = ['retrain']#'static', 'train','retrain',],'static'
cnns_pretrained = dict(VGG19 = VGG19)#Xception = Xception, ,InceptionV3 =InceptionV3)#VGG19 = VGG19)#, Xception = Xception, ResNet50 = ResNet50)#, InceptionV3 =InceptionV3)
lstms_type = dict(
    COV_LSTM = (ConvLSTM2D, dict(filters=256, kernel_size=(3, 3),padding='same', return_sequences=False))
                  )

results = []
for dataset_name, dataset_videos in datasets_videos.items():
    train_path, valid_path, test_path, \
    train_y, valid_y, test_y, \
    avg_length = DatasetBuilder.createDataset(dataset_videos, datasets_frames,fix_len, force = force)
    if fix_len is not None:
        avg_length = fix_len

    train_gen = DatasetBuilder.data_generator(train_path, train_y, batch_size, figure_size, avg_length)
    validate_gen = DatasetBuilder.data_generator(valid_path, valid_y, batch_size, figure_size, avg_length)
    # train_X_, train_y_ = DatasetBuilder.get_sequences(train_path, train_y, figure_size, avg_length)
    # train_gen = DatasetBuilder.data_generator_files(train_X_, train_y_, batch_size)
    # valid_X_, valid_y_ = DatasetBuilder.get_sequences(valid_path, valid_y, figure_size, avg_length)
    # validate_gen = DatasetBuilder.data_generator_files(valid_X_, valid_y_, batch_size)

    test_x, test_y = DatasetBuilder.get_sequences(test_path, test_y, figure_size, avg_length, )
    for learning_rate in learning_rates:
        for cnn_train_type in cnn_train_types:
            for cnn_name, cnn_class in cnns_pretrained.items():
                for lstm_name, lstm_conf in lstms_type.items():
                    result = dict(dataset = dataset_name, cnn_train = cnn_train_type,cnn = cnn_name, lstm = lstm_name,epochs = epochs,learning_rate = learning_rate, batch_size = batch_size,
                                             optimizer = optimizer[0].__class__.__name__, initial_weights = initial_weights,seq_len = avg_length)
                    print("run experimnt " + str(result))
                    model = BuildModel_basic.build(size = figure_size, seq_len = avg_length, learning_rate = learning_rate,
                                                   optimizer_class = optimizer, initial_weights = initial_weights,
                                             cnn_class = cnn_class,pre_weights = weights, lstm_conf = lstm_conf,cnn_train_type = cnn_train_type, classes= classes)
                    # if load_all:
                    #     model.fit(train_x, train_y, epochs=epochs,batch_size = batch_size)
                    # else:
                    history = model.fit_generator(
                        steps_per_epoch = int(float(len(train_path)) / float(batch_size*4.)),
                        generator=train_gen,
                        epochs=epochs,
                        validation_data=validate_gen,
                        validation_steps= len(valid_path) / batch_size
                        )

                    model_name = ""
                    for k, v in result.items():
                        model_name  = model_name + "_" + str(k) + "-" + str(v).replace(".", "d")
                    model_path = os.path.join(res_path, model_name)
                    pd.DataFrame(history.history).to_csv(model_path + "_train_results.csv")
                    evals = model.evaluate(test_x,test_y,batch_size=batch_size)
                    result['test loss'] = evals[0]
                    result['test accuracy'] = evals[1]
                    results.append(result)
                    pd.DataFrame(results).to_csv("results.csv")
                    print(result)
pd.DataFrame(results).to_csv("results.csv")

# def hyper_tune_network(dataset_name, epochs, batch_size, batch_epoch_ratio, figure_size, initial_weights, lstm, cnns_arch,
#                        learning_rates, optimizers, cnn_train_types):
#     results = []
#     train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(datasets_videos[dataset_name], datasets_frames, fix_len,
#                                                                       figure_size, force=force)
#     hyper = dict()
#     best_accuracy = 0.5
#
#     exp_params = dict(dataset_name=dataset_name, train_gen=train_gen,
#                         validate_gen=validate_gen, test_x=test_x, test_y=test_y, seq_len=seq_len, epochs=epochs,
#                         batch_size=batch_size, batch_epoch_ratio=batch_epoch_ratio, initial_weights=initial_weights, size=figure_size,
#                       len_train = len_train, len_valid = len_valid)
#
#     for optimizer in optimizers:
#         hyper['optimizer'] = optimizer
#         if 'learning_rate' in hyper:
#             learning_rate = hyper['learning_rate']
#             cnn_train_type = hyper['cnn_train_type']
#             cnn_arch = hyper['cnn_arch']
#             result = train_eval_network(cnn_arch=cnn_arch, learning_rate=learning_rate,
#                                         optimizer=optimizer, cnn_train_type=cnn_train_type,
#                                         pre_weights=weights, lstm_conf=lstm, **exp_params)
#             print(result)
#             results.append(result)
#             if result['test accuracy'] > best_accuracy:
#                 best_accuracy = result['test accuracy']
#                 hyper['optimizer'] = optimizer
#         else:
#             hyper['learning_rate'] = learning_rates[0]
#             for learning_rate in learning_rates:
#                 if 'cnn_train_type' in hyper:
#                     cnn_train_type = hyper['cnn_train_type']
#                     cnn_arch = hyper['cnn_arch']
#                     result = train_eval_network(cnn_arch=cnn_arch, learning_rate=learning_rate,
#                                                 optimizer=optimizer, cnn_train_type=cnn_train_type,
#                                                 pre_weights=weights, lstm_conf=lstm, **exp_params)
#                     print(result)
#                     results.append(result)
#                     if result['test accuracy'] > best_accuracy:
#                         best_accuracy = result['test accuracy']
#                         hyper['learning_rate'] = learning_rate
#                 else:
#                     hyper['cnn_train_type'] = cnn_train_types[0]
#                     for cnn_train_type in cnn_train_types:
#                         if 'cnn_arch' in hyper:
#                             cnn_arch = hyper['cnn_arch']
#                             result = train_eval_network(cnn_arch=cnn_arch, learning_rate=learning_rate,
#                                                         optimizer=optimizer, cnn_train_type=cnn_train_type,
#                                                         pre_weights=weights, lstm_conf=lstm, **exp_params)
#                             results.append(result)
#                             if result['test accuracy'] > best_accuracy:
#                                 best_accuracy = result['test accuracy']
#                                 hyper['cnn_train_type'] = cnn_train_type
#                         else:
#                             for cnn_arch_name,cnn_arch in cnns_arch.items():
#                                 result = train_eval_network(cnn_arch=cnn_arch, learning_rate=learning_rate,
#                                                             optimizer=optimizer, cnn_train_type=cnn_train_type,
#                                                             pre_weights=weights, lstm_conf=lstm, **exp_params)
#                                 results.append(result)
#                                 if result['test accuracy'] > best_accuracy:
#                                     best_accuracy = result['test accuracy']
#                                     hyper['cnn_arch'] = cnn_arch
#
#     return hyper, results