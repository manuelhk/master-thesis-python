import cv2
import numpy as np
import glob
import random


IMAGE_SHAPE = (224, 224)
INPUT_DIRECTORY = "nn/data"
OUTPUT_DIRECTORY = "nn/data"


def create_npy_from_videos(videos_directory):
    videos = glob.glob(videos_directory + "/*")
    videos.sort()
    list = []
    for video_path in videos:
        list.append(load_video(video_path))
    videos_np = np.array(list)
    return videos_np


def create_npy_from_images(images_directory):
    images = glob.glob(images_directory + "/*")
    images.sort()
    list = []
    for image_path in images:
        list.append(load_image(image_path))
    images_np = np.array(list)
    return images_np


def load_video(video_path):
    images = glob.glob(video_path + "/*.jpg")
    images.sort()
    list = []
    for image_path in images:
        image = load_image(image_path)
        list.append(image)
    video_np = np.array(list)
    return video_np


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SHAPE)
    image_np = np.array(image)
    return image_np


def jpg_to_npy(directory):
    images = glob.glob(directory + "*.jpg")
    images.sort()
    i = 0
    for path in images:
        np.save(directory + str(i) + ".npy", load_image(path))
        i += 1
    pass


def get_data_and_labels(directory):
    paths = glob.glob(directory + "*/*.npy")
    paths.sort()
    labels_dict = dict()
    for path in paths:  # todo create method to label data correctly
        if "left" in path:
            labels_dict.update({path: 0})
        else:
            labels_dict.update({path: 1})
    # labels = [0 if "left" in path else 1 for path in paths]
    data_dict = split_into_training_and_validation(paths)
    return data_dict, labels_dict


def split_into_training_and_validation(list_of_paths, share_training_data=0.8):
    random.shuffle(list_of_paths)
    train_list = list_of_paths[:np.math.floor(len(list_of_paths)*share_training_data)]
    eval_list = list_of_paths[np.math.floor(len(list_of_paths)*share_training_data):]
    data_dict = {"train": train_list, "validation": eval_list}
    return data_dict


directory = "nn/data/"
# jpg_to_npy(directory)

d, l = get_data_and_labels(directory)
t = d["train"]
for el in t:
    label = l[el]
    print(el + ": " + str(label))

