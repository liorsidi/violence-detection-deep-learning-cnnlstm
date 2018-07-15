import scipy
import os
import cv2
import pickle
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.preprocessing import image

def save_figures_from_video(dataset_video_path, video_filename,figures_path,skip_frames = 25):
    seq_len = 0

    video_figures_path = os.path.join(figures_path ,video_filename)
    if not os.path.exists(video_figures_path):
        os.makedirs(video_figures_path)
    video_figures_path = video_figures_path + 'frame_'

    video_file = os.path.join(dataset_video_path, video_filename + ".avi")
    label = 0
    print('Extracting frames from video: ', video_file)
    videoCapture = cv2.VideoCapture(video_file)
    success, figure = videoCapture.read()
    success = True
    files = []
    while success:
        videoCapture.set(cv2.CAP_PROP_POS_MSEC, (seq_len * skip_frames))
        success, figure = videoCapture.read()
        if success:
            image_file = video_figures_path + "%d.jpg" % seq_len
            files.append(image_file)
            cv2.imwrite(image_file, figure)
        seq_len += 1
    video_images = dict(images_path = video_figures_path, name = video_filename,
                        images_files = files, sequence_length = seq_len, label = label)

    return video_images

def createDataset(datasets_video_path, figure_output_path, figure_size, split_ratio, force = True):
    videos_seq_length = []
    datasets_images = {}
    videos_frames_paths = []
    videos_labels = []
    #Extract images for each video for each dataset
    for dataset_name, dataset_video_path in datasets_video_path.iteritems():
        dataset_figures_path = os.path.join(figure_output_path,dataset_name)
        if not os.path.exists(dataset_figures_path):
            os.makedirs(dataset_figures_path)
        dataset_images = []
        for filename in os.listdir(dataset_video_path):
            if filename.endswith(".avi"):
                video_images_file = os.path.join(dataset_figures_path,filename[:-4], 'video_summary.pkl')
                if os.path.isfile(video_images_file) and not force:
                    with open(video_images_file, 'rb') as f:
                        video_images = pickle.load(f)
                else:
                    video_images = save_figures_from_video(dataset_video_path, filename[:-4], dataset_figures_path)
                    if dataset_name == "hocky":
                        if filename.startswith("fi"):
                            video_images['label'] = 1
                    elif dataset_name == "violentflow":
                        if not dataset_video_path.contains("Non"):
                            video_images['label'] = 1
                    elif dataset_name == "movies":
                        if not dataset_video_path.contains("noFights"):
                            video_images['label'] = 1
                    with open(video_images_file, 'wb') as f:
                        pickle.dump(video_images, f, pickle.HIGHEST_PROTOCOL)
                dataset_images.append(video_images)
                videos_seq_length.append(video_images['sequence_length'])
                videos_frames_paths.append(video_images['images_path'])
                videos_labels.append(video_images['label'])
        datasets_images[dataset_name] = dataset_images
    avg_length = int(float(sum(videos_seq_length)) / max(len(videos_seq_length), 1))

    # sequences_paths = []
    # for dataset_name, dataset_images in datasets_images.iteritems():
    #     for filename in os.listdir(dataset_images['images_path']):
    #         frames = sorted(glob.glob(os.path.join(figure_output_path,dataset_name + '*jpg')))
    #         #TODO resize
    #         #TODO transformation
    #         #TODO compute diff
    #         #TODO save as a sequence
    #         #TODO add ti dict

    train_path, test_path =  train_test_split(videos_frames_paths, test_size=0.20, random_state=42)
    train_y, test_y = train_test_split(videos_labels, test_size=0.20, random_state=42)

    return train_path, test_path,train_y, test_y, avg_length

def frame_loader(frames,figure_shape):
    output_frames = []
    for frame in frames:
        image = load_img(frame, target_size=(figure_shape, figure_shape))
        img_arr = img_to_array(image)
        x = (img_arr / 255.).astype(np.float32)
        output_frames.append(x)
    return output_frames


def data_generator(data_paths,labels,batch_size,figure_shape,seq_length):
    while True:
        X, y = [], []
        indexes = np.arange(len(data_paths))
        np.random.shuffle(indexes)
        select_indexes = indexes[:batch_size]
        for select_index in select_indexes:
            frames = sorted(glob.glob(os.path.join(data_paths[select_index], '*jpg')))
            x = frame_loader(frames, figure_shape)
            X.append(x)
            y.append(labels[select_index])
        X = pad_sequences(X,maxlen = seq_length, padding = 'pre' , truncating = 'pre' )
        yield np.array(X), np.array(y)

def load_data(data_paths,labels,figure_shape,seq_length):
    X, y = [], []
    for select_index in range(len(data_paths)):
        frames = sorted(glob.glob(os.path.join(data_paths[select_index]+ '*jpg')))
        x = frame_loader(frames, figure_shape)
        X.append(x)
        y.append(labels[select_index])
    X = pad_sequences(X,maxlen = seq_length, padding = 'pre' , truncating = 'pre' )
    return np.array(X), np.array(y)