import numpy as np
import matplotlib.pyplot as plt
import glob


def show_npy(array, number_of_images):
    # array = np.load(path)
    for i in range(0, 15, int(15/number_of_images)):
        plt.imshow(array[i])
        plt.xlabel(path + " (" + str(i) + ")")
        plt.show()
    pass


# l = glob.glob("test/*/*.npy")
# show_npy('test/FREE_CRUISING/FREE_CRUISING_9.npy', 15)
