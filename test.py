import my_model
import numpy as np
import glob


SCENARIOS = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]


# model = keras.applications.VGG16()
# keras.utils.plot_model(model, to_file='model.png')

# model = my_model.build_image_model(5)
# print(model.summary())


def videos_to_images(input_dir, output_dir):
    for scenario in SCENARIOS:
        video_paths = glob.glob(input_dir + "/" + scenario + "/*.npy")
        print(str(len(video_paths)) + " videos of scenario " + scenario)
        count_video = 0
        for path in video_paths:
            video = np.load(path)
            count_frame = 0
            for i in range(15):
                np.save(output_dir + "/" + scenario + "/" + scenario +
                        "_v" + str(count_video) + "_f" + str(count_frame) + ".npy", video[i])
                count_frame += 1
            count_video += 1
    pass
