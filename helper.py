import cv2
import numpy as np


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
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        count += 1
        array = np.array(img)
        array = array[:, 106:746, :]
        a.append(array)
        if count % 15 == 0:
            np.save(output_path + str(count) + ".npy", np.array(a))
            a = []
    pass


# video_to_jpges_and_npys("data/video.avi", "data/video/")
