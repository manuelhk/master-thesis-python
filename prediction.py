import keras
import numpy as np
import os
import helper
import glob


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


def predict_and_show(model, scenario_paths, labels):
    for path in scenario_paths:
        pred = predict(model, path)
        title = ""
        for i in range(len(labels)):
            title = title + "  |  " + str(pred[i]) + " - " + labels[i]
        helper.show_npy(path, title=title)
    pass
