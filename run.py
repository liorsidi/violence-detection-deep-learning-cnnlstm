raw_data = dict(hocky = "data/hocky")
dataset_path = "/datasets/"
figure_size = 256
split_ratio = 0.8
train_path, test_path, avg_length = DatasetBuilder.createDataset(raw_data, dataset_path + raw_name, figure_size,split_ratio)
train_x, train_y, test_x, test_y = DatasetBuilder.loadData(train_path, test_path )

epoch = 10
learning_rate = 0.0004
batch_size = 16
optimizer ='RMSprop'
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
            #lstm = lstm_conf[0](**lstm_conf[1])
            result = dict(cnn_train = cnn_train_type,cnn = cnn_name, lstm = lstm_name,epoch = epoch,learning_rate = learning_rate, batch_size = batch_size,
                                     optimizer = optimizer, initial_weights = initial_weights)
            model = BuildModel.build(epoch = epoch,learning_rate = learning_rate, batch_size = batch_size,
                                     optimizer = optimizer, initial_weights = initial_weights,
                                     cnn_class = cnn_class,pre_weights = weights, lstm_conf = lstm_conf)
            model.fit(train_x,train_y)
            preds = model.predcit(test_x)
            result['accuracy'] = accuracy_score(preds, test_y)
            results.append(result)
            print result
pd.DataFrame(results).to_csv("results.csv")