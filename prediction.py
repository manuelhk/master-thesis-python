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


def predict_and_print(model, scenario_paths, lables):
    for path in scenario_paths:
        path_list = path.split(os.sep)
        pred = predict(model, path)
        print("-----------------------------------")
        print("Actual: " + path_list[-2])
        for i in range(len(lables)):
            print(str(pred[i]) + " - " + lables[i])
    pass


def predict_and_show(model, scenario_paths, lables):
    for path in scenario_paths:
        pred = predict(model, path)
        title = ""
        for i in range(len(lables)):
            title = title + "  |  " + str(pred[i]) + " - " + lables[i]
        helper.show_npy(path, title=title)
    pass


scs = ["free_cruising", "following", "catching_up"]
model = keras.models.load_model("data/v3_lstm_fr_fo_ca.h5")
paths = glob.glob("data/video/*.npy")
