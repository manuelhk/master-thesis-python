import numpy as np
import matplotlib.pyplot as plt
import cv2

def video_to_jpges_and_npys(video_path, output_path):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    a = []
    while success:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(output_path + "frame%d.jpg" % count, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:, 106:746, :]
        image = cv2.resize(image, (299, 299))
        count += 1
        array = np.array(image)
        a.append(array)
        if count % 15 == 0:
            np.save(output_path + str(count) + ".npy", np.array(a))
            a = []
    pass


def show_npy(path, number_of_images=3, title="No Title"):
    array = np.load(path)
    for i in range(0, 15, int(15/number_of_images)):
        plt.imshow(array[i])
        plt.xlabel(path + " (" + str(i) + ")")
        plt.title(title)
        plt.show()
    pass


# l = glob.glob("test/*/*.npy")
# show_npy('test/FREE_CRUISING/FREE_CRUISING_9.npy', 15)

# video_to_jpges_and_npys("data/video.avi", "data/video/")
