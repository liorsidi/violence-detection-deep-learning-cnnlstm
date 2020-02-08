# Learning to Detect Violent Videos using Convolution LSTM

This work is based on violence detection model proposed by [1] with minor modications.
The original model was implemented with Pytorch [2] while in this work we implement it with Keras and TensorFlow as a back-end. 
The model incorporates pre-trained convolution Neural Network (CNN) connected to Convolutional LSTM (ConvLSTM) layer.
The model takes as an inputs the raw video, converts it into frames and output a binary classication of violence or non-violence label.

### Architecture structure
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/Architecture.jpeg)


## Running configurations
### Video datasets paths:
data path are defined as follows:
- hocky - data/raw_videos/HockeyFights - [Data_Link](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89/tech)
- violentflow - data/raw_videos/violentflow - [Data_Link](https://www.openu.ac.il/home/hassner/data/violentflows/)
- movies - data/raw_videos/movies - [Data_Link](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)

### Libraries perquisites:
- python 2.7
- numpy 1.14.0
- keras 2.2.0
- tensorflow 1.9.0
- Pillow 3.1.2
- opencv-python 3.4.1.15

### Running operation:
just run python run.py
(currently we don't support arguments from command line)

## Results
#### Hyper-tuning results (Hocky data)
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/hyperparameters_results.JPG)

#### Hockey dataset results
![alt text](https://github.com/liorsidi/ViolenceDetection_CNNLSTM/blob/master/images/Hockey_results.png)

## Refrences
1. Sudhakaran, Swathikiran, and Oswald Lanz. "Learning to detect violent videos
using convolution long short-term memory." In Advanced Video and Signal Based
Surveillance (AVSS), 2017 14th IEEE International Conference on, pp. 1-6. IEEE, 2017.
2. https://github.com/swathikirans/violence-recognition-pytorch
