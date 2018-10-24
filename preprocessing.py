import cv2
import numpy as np
import glob


def load_image(image_path):
    """ Loads image and returns numpy of image"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np = np.array(image)
    return image_np


def load_video(images):
    """ Loads images in defined folder (from one video) and returns numpy of video """
    img_list = []
    for image_path in images:
        image = load_image(image_path)
        img_list.append(image)
    video_np = np.array(img_list)
    return video_np


def split_label(label_np, min_consecutive_scenes):
    start_indices = []
    rows = label_np.__len__()
    for i in range(rows):
        if np.sum(label_np[i:i+min_consecutive_scenes]) == min_consecutive_scenes:
            start_indices.append(i)
    return start_indices


def prepare_images(label_np, images, scenarios, min_consecutive_scenes, out_path):
    print("Save numpys...")
    no_labels = scenarios.__len__()
    for i in range(no_labels):
        label_dir = out_path + "/" + scenarios[i]
        no_existing_files = glob.glob(label_dir + "/*").__len__()
        start_indices = split_label(label_np[:, i], min_consecutive_scenes)
        for index in start_indices:
            video_np = load_video(images[index: index+min_consecutive_scenes])
            np.save(label_dir + "/" + str(no_existing_files), video_np)
            no_existing_files += 1
        print(scenarios[i] + ": " + str(start_indices.__len__()) + " new video arrays")
    pass
