import cv2
import numpy as np
import glob
import random


IMAGE_SHAPE = (150, 150)
ROOT_DIRECTORY = "data/"


def load_video(video_path):
    """ Loads images in defined folder (from one video) and returns numpy of video """
    images = glob.glob(video_path + "/*.jpg")
    images.sort()
    list = []
    for image_path in images:
        image = load_image(image_path)
        image = cv2.resize(image, IMAGE_SHAPE)
        list.append(image)
    video_np = np.array(list)
    return video_np


def load_image(image_path):
    """ Loads image and returns numpy of image"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SHAPE)
    image_np = np.array(image)
    return image_np


def videos_to_npy(directory):
    """ Accesses every video folder in directory and saves a .npy file respectively"""
    videos = glob.glob(directory + "*")
    i = 0
    for path in videos:
        np.save(directory + str(i) + ".npy", load_video(path))
        i += 1
    pass


def images_to_npy(directory):
    """ Accesses every image in directory and creates and saves a .npy file respectively """
    images = glob.glob(directory + "*.jpg")
    images.sort()
    i = 0
    for path in images:
        np.save(directory + str(i) + ".npy", load_image(path))
        i += 1
    pass


def get_data_and_labels(directory):
    """ Accesses directory and creates two dictionaries with filenames .npy. One for labels and the other for data. """
    paths = glob.glob(directory + "*/*.npy")
    # paths.sort()
    labels_dict = dict()
    for path in paths:  # todo create method to label data correctly
        if "label0" in path:
            labels_dict.update({path: 0})
        else:
            labels_dict.update({path: 1})
    # labels = [0 if "left" in path else 1 for path in paths]
    data_dict = split_into_training_and_validation(paths)
    return data_dict, labels_dict


def split_into_training_and_validation(list_of_paths, share_training_data=0.8):
    """ Splits data into training data and validation data and returns a dictionary with two lists """
    random.shuffle(list_of_paths)
    train_list = list_of_paths[:np.math.floor(len(list_of_paths)*share_training_data)]
    eval_list = list_of_paths[np.math.floor(len(list_of_paths)*share_training_data):]
    data_dict = {"train": train_list, "validation": eval_list}
    return data_dict


dir = ROOT_DIRECTORY + "test/"
# video_paths = glob.glob(dir + "*")
