import keras
import numpy as np
import os
import glob
import random


def show_results(model, scenario_paths, labels):
    for path in scenario_paths:
        predict_and_print(model, path, labels)
    pass


def predict_and_print(model, path, lables):
    path_list = path.split(os.sep)
    video = np.load(path)
    video = np.expand_dims(video, 0)
    video = keras.applications.inception_v3.preprocess_input(video)
    pred = model.predict(video)
    pred_rounded = np.around(pred[0], 4)
    print("-----------------------------------")
    print("Actual: " + path_list[-2])
    for i in range(len(lables)):
        print(str(pred_rounded[i]) + " - " + lables[i])
    pass

scs = ["free_cruising", "following", "catching_up"]
model = keras.models.load_model("training/v3_lstm_fr_fo_ca.h5")
paths = glob.glob("output/*/*.npy")
random.shuffle(paths)
