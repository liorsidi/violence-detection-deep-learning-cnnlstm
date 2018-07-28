# Learning to Detect Violent Videos using Convolution LSTM

This work is based on violence detection model proposed by [1] with minor modications.
The original model was implemented with Pytorch [2] while in this work we implement it with Keras and TensorFlow as a back-end. 
The model incorporates pre-trained convolution Neural Network (CNN) connected to Convolutional LSTM (ConvLSTM) layer.
The model takes as an inputs the raw video, converts it into frames and output a binary classication of violence or non-violence label.

#### Architecture stracture
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/Architecture.jpeg)

#### Hypertunning results
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/hyperparameters_results.JPG)

#### Hockey dataset results
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/Hockey_results.png)

## Refrences
1. Sudhakaran, Swathikiran, and Oswald Lanz. "Learning to detect violent videos
using convolution long short-term memory." In Advanced Video and Signal Based
Surveillance (AVSS), 2017 14th IEEE International Conference on, pp. 1-6. IEEE, 2017.
2. https://github.com/swathikirans/violence-recognition-pytorch
