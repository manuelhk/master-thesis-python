<<<<<<< HEAD
import my_generator


DIRECTORY = "output"
SCENARIOS = ["free_cruising", "following", "overtaking"]
PARAMS = {'dim': (15, 150, 150),
          'batch_size': 5,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}
=======
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
>>>>>>> 5d5cc450bfac10d0cf7649035f67de51a9751ff6
