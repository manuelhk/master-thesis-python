import cv2
import numpy as np
import glob


def load_video(video_path):
    """ Loads images in defined folder (from one video) and returns numpy of video """
    images = glob.glob(video_path + "/*.jpg")
    images.sort()
    img_list = []
    for image_path in images:
        image = load_image(image_path)
        image = cv2.resize(image, (150, 150))
        img_list.append(image)
    video_np = np.array(img_list)
    return video_np


def load_image(image_path):
    """ Loads image and returns numpy of image"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
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
