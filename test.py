import numpy as np
import glob


def split_npy(array_path):
    np_array = np.load(array_path)
    frames, w, h, c = np_array.shape
    for i in range(frames):
        np.save(array_path + "_" + str(i) + ".npy", np_array[i])
    pass


path = "test2/free_cruising"
videos = glob.glob(path + "/*.npy")
for video in videos:
    split_npy(video)
