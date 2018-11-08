import keras
import glob
import numpy as np
import os
import random


def predict(model, scenario_path):
    video = np.load(scenario_path)
    video = np.expand_dims(video, 0)
    video = keras.applications.inception_v3.preprocess_input(video)
    prediction = model.predict(video)
    pred_rounded = np.around(prediction[0], 4)
    return pred_rounded


def predict_and_print(model, scenario_paths, labels):
    for path in scenario_paths:
        path_list = path.split(os.sep)
        pred = predict(model, path)
        print("-----------------------------------")
        print("Actual: " + path_list[-2])
        for i in range(len(labels)):
            print(str(pred[i]) + " - " + labels[i])
    pass


labels = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]
scenario_paths = glob.glob("input/*/*.npy")
random.shuffle(scenario_paths)

model = keras.models.load_model("output/1107_v3_lstm_fr_fo_ca_lcl_lcr_950_50.h5")

predict_and_print(model, scenario_paths[0:50], labels)
