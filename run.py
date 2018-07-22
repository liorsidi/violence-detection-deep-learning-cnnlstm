import os

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam

import pandas as pd
from keras.applications import Xception, ResNet50, InceptionV3, MobileNet, VGG19, DenseNet121, InceptionResNetV2, VGG16
from keras.layers import LSTM, ConvLSTM2D
import BuildModel_basic
import DatasetBuilder

from numpy.random import seed

from tensorflow import set_random_seed

def train_eval_network(dataset_name ,train_gen ,validate_gen ,test_x, test_y , seq_len , epochs, batch_size, batch_epoch_ratio, initial_weights, size, cnn_arch, learning_rate,
                       optimizer, cnn_train_type, pre_weights, lstm_conf, len_train, len_valid):
        set_random_seed(2)
        seed(1)
        result = dict(dataset=dataset_name, cnn_train=cnn_train_type,
                      cnn=cnn_arch.__name__, lstm=lstm_conf[0].__name__, epochs=epochs,
                      learning_rate=learning_rate, batch_size=batch_size,
                      optimizer=optimizer[0].__name__, initial_weights=initial_weights, seq_len=seq_len)
        print("run experimnt " + str(result))
        model = BuildModel_basic.build(size=size, seq_len=seq_len, learning_rate=learning_rate,
                                       optimizer_class=optimizer, initial_weights=initial_weights,
                                       cnn_class=cnn_arch, pre_weights=pre_weights, lstm_conf=lstm_conf,
                                       cnn_train_type=cnn_train_type)
        history = model.fit_generator(
            steps_per_epoch=int(float(len_train) / float(batch_size * batch_epoch_ratio)),
            generator=train_gen,
            epochs=epochs,
            validation_data=validate_gen,
            validation_steps= int(float(len_valid) / float(batch_size)),
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.1, patience=30,),
                         ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        ]
        )

        model_name = ""
        for k, v in result.items():
            model_name = model_name + "_" + str(k) + "-" + str(v).replace(".", "d")
        model_path = os.path.join(res_path, model_name)
        pd.DataFrame(history.history).to_csv(model_path + "_train_results.csv")
        evals = model.evaluate(test_x, test_y, batch_size=batch_size)
        result['test loss'] = evals[0]
        result['test accuracy'] = evals[1]
        return result


def get_generators(dataset_videos, datasets_frames, fix_len, figure_size, force):
    train_path, valid_path, test_path, \
    train_y, valid_y, test_y, \
    avg_length = DatasetBuilder.createDataset(dataset_videos, datasets_frames, fix_len, force=force)
    if fix_len is not None:
        avg_length = fix_len

    len_train, len_valid = len(train_path), len(valid_path)
    train_gen = DatasetBuilder.data_generator(train_path, train_y, batch_size, figure_size, avg_length)
    validate_gen = DatasetBuilder.data_generator(valid_path, valid_y, batch_size, figure_size, avg_length)
    test_x, test_y = DatasetBuilder.get_sequences(test_path, test_y, figure_size, avg_length)

    return train_gen, validate_gen, test_x, test_y, avg_length, len_train, len_valid


def hyper_tune_network_(dataset_name, epochs, batch_size, batch_epoch_ratio, figure_size, initial_weights, lstm, cnns_arch,
                       learning_rates, optimizers, cnn_train_types):
    results = []
    train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(datasets_videos[dataset_name], datasets_frames, fix_len,
                                                                      figure_size, force=force)

    best_accuracy = 0.5


    params_to_train = dict(dataset_name=dataset_name, train_gen=train_gen,
                        validate_gen=validate_gen, test_x=test_x, test_y=test_y, seq_len=seq_len, epochs=epochs,
                        batch_size=batch_size, batch_epoch_ratio=batch_epoch_ratio, initial_weights=initial_weights, size=figure_size,
                      len_train = len_train, len_valid = len_valid,  pre_weights=weights, lstm_conf = lstm)

    exp_params_order = ['cnn_arch','learning_rate','optimizer','cnn_train_type']
    exp_params = dict(optimizer= optimizers, learning_rate= learning_rates, cnn_train_type = cnn_train_types, cnn_arch = cnns_arch.values())
    best_params = dict(optimizer=optimizers[0], learning_rate=learning_rates[0], cnn_train_type=cnn_train_types[0],cnn_arch=cnns_arch.values()[0])

    for exp_param in exp_params_order:
        temp_param = dict(best_params)
        for param in exp_params[exp_param]:
            temp_param[exp_param] = param
            params_to_train.update(temp_param)
            result = train_eval_network(**params_to_train)
            results.append(result)
            if result['test accuracy'] > best_accuracy:
                best_accuracy = result['test accuracy']
                best_params[exp_param] = param
                print("best accuracy update " + str(best_accuracy))
    return best_params, results


def hyper_tune_network(dataset_name, epochs, batch_size, batch_epoch_ratio, figure_size, initial_weights, lstm, cnns_arch,
                       learning_rates, optimizers, cnn_train_types):
    results = []
    train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(datasets_videos[dataset_name], datasets_frames, fix_len,
                                                                      figure_size, force=force)
    hyper = dict()
    best_accuracy = 0.5

    exp_params = dict(dataset_name=dataset_name, train_gen=train_gen,
                        validate_gen=validate_gen, test_x=test_x, test_y=test_y, seq_len=seq_len, epochs=epochs,
                        batch_size=batch_size, batch_epoch_ratio=batch_epoch_ratio, initial_weights=initial_weights, size=figure_size,
                      len_train = len_train, len_valid = len_valid)

    for optimizer in optimizers:
        hyper['optimizer'] = optimizer
        if 'learning_rate' in hyper:
            learning_rate = hyper['learning_rate']
            cnn_train_type = hyper['cnn_train_type']
            cnn_arch = hyper['cnn_arch']
            result = train_eval_network(cnn_arch=cnn_arch, learning_rate=learning_rate,
                                        optimizer=optimizer, cnn_train_type=cnn_train_type,
                                        pre_weights=weights, lstm_conf=lstm, **exp_params)
            results.append(result)
            if result['test accuracy'] > best_accuracy:
                best_accuracy = result['test accuracy']
                hyper['optimizer'] = optimizer
        else:
            hyper['learning_rate'] = learning_rates[0]
            for learning_rate in learning_rates:
                if 'cnn_train_type' in hyper:
                    cnn_train_type = hyper['cnn_train_type']
                    cnn_arch = hyper['cnn_arch']
                    result = train_eval_network(cnn_arch=cnn_arch, learning_rate=learning_rate,
                                                optimizer=optimizer, cnn_train_type=cnn_train_type,
                                                pre_weights=weights, lstm_conf=lstm, **exp_params)
                    results.append(result)
                    if result['test accuracy'] > best_accuracy:
                        best_accuracy = result['test accuracy']
                        hyper['learning_rate'] = learning_rate
                else:
                    hyper['cnn_train_type'] = cnn_train_types[0]
                    for cnn_train_type in cnn_train_types:
                        if 'cnn_arch' in hyper:
                            cnn_arch = hyper['cnn_arch']
                            result = train_eval_network(cnn_arch=cnn_arch, learning_rate=learning_rate,
                                                        optimizer=optimizer, cnn_train_type=cnn_train_type,
                                                        pre_weights=weights, lstm_conf=lstm, **exp_params)
                            results.append(result)
                            if result['test accuracy'] > best_accuracy:
                                best_accuracy = result['test accuracy']
                                hyper['cnn_train_type'] = cnn_train_type
                        else:
                            for cnn_arch_name,cnn_arch in cnns_arch.items():
                                result = train_eval_network(cnn_arch=cnn_arch, learning_rate=learning_rate,
                                                            optimizer=optimizer, cnn_train_type=cnn_train_type,
                                                            pre_weights=weights, lstm_conf=lstm, **exp_params)
                                results.append(result)
                                if result['test accuracy'] > best_accuracy:
                                    best_accuracy = result['test accuracy']
                                    hyper['cnn_arch'] = cnn_arch

    return hyper, results


datasets_videos = dict(hocky = dict(hocky ="data/raw_videos/HockeyFights"),
                       violentflow=dict(violentflow="data/raw_videos/violentflow"),
                       movies=dict(movies="data/raw_videos/movies"))
datasets_frames = "data/raw_frames"
res_path = "results"
figure_size = 244
split_ratio = 0.8
batch_size = 2
batch_epoch_ratio = 4.
fix_len = 20
initial_weights = 'glorot_uniform'
weights='imagenet'
force = False

optimizers =[(RMSprop,{}), (Adam, {})]
learning_rates = [1e-3, 1e-4,1e-5, ] #1e-4, 1e-6
cnn_train_types = ['retrain','static'] #'retrain',],'static'
cnns_arch = dict(ResNet50 = ResNet50, VGG16 = VGG16,InceptionV3 =InceptionV3, VGG19 = VGG19,)  #,InceptionV3 =InceptionV3, VGG19 = VGG19
#Xception = Xception, ,InceptionV3 =InceptionV3)#VGG19 = VGG19)#, Xception = Xception, ResNet50 = ResNet50)#, InceptionV3 =InceptionV3)
# Too big
# Xception = Xception
# DenseNet121 = DenseNet121
# ResNet50 = ResNet50
# conf error
#InceptionResNetV2= InceptionResNetV2
# MobileNet = MobileNet
# InceptionV3 =InceptionV3,
# VGG19 = VGG19
# VGG16 = VGG16

lstm = (ConvLSTM2D, dict(filters=256, kernel_size=(3, 3),padding='same', return_sequences=False))
apply_hyper = True
if apply_hyper:
    hyper, results= hyper_tune_network_(dataset_name = 'hocky', epochs = 5,
                           batch_size = batch_size, batch_epoch_ratio = batch_epoch_ratio,figure_size = figure_size,
                           initial_weights = initial_weights, lstm = lstm,
                           cnns_arch = cnns_arch, learning_rates = learning_rates,
                           optimizers = optimizers, cnn_train_types = cnn_train_types)

    pd.DataFrame(results).to_csv("hyper_results_3.csv")

    cnn_arch, learning_rate,optimizer, cnn_train_type = hyper['cnn_arch'],\
                                                        hyper['learning_rate'],\
                                                        hyper['optimizer'],\
                                                        hyper['cnn_train_type'],
else:
    results = []
    cnn_arch, learning_rate,optimizer, cnn_train_type = ResNet50, 0.0001, (RMSprop,dict(decay=0.5)), 'retrain'
for dataset_name, dataset_videos in datasets_videos.items():
    train_gen, validate_gen, test_x, test_y, seq_len, len_train, len_valid = get_generators(dataset_videos, datasets_frames,fix_len,figure_size, force = force)
    result = train_eval_network(epochs = 500, dataset_name = dataset_name,train_gen = train_gen,validate_gen = validate_gen,
                                test_x= test_x, test_y = test_y, seq_len = seq_len,batch_size = batch_size,
                                batch_epoch_ratio = batch_epoch_ratio,initial_weights = initial_weights,size = figure_size,
                                cnn_arch = cnn_arch, learning_rate = learning_rate,
                                optimizer = optimizer, cnn_train_type = cnn_train_type,
                                pre_weights = weights, lstm_conf = lstm, len_train = len_train, len_valid = len_valid)
    results.append(result)
    pd.DataFrame(results).to_csv("results_3.csv")
    print(result)
pd.DataFrame(results).to_csv("results_3.csv")

