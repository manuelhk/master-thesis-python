import my_model
import my_generator
import keras
import random
import numpy as np


################################################################################
################################################################################

# This script is used to design neural networks for video classification and train
# them with video data (in form of numpy arrays)
#
# With this script it is possible to mix synthetic and real input data with different ratios

################################################################################
################################################################################


input_directory_sim = "input"       # directory where all synthetic input data is stored in respective class subfolders
input_directory_real = "input/real" # directory where all real input data is stored in respective class subfolders
output_directory = "output"         # directory where the trained model is stored

classification = "video"            # based video or image classification (with or without LSTM-layer)
cnn_name = "inception_v3"           # inception_v3 or xception
dropout = False                     # determines whether or not dropout is applied to the second to last layer
dim = (15, 299, 299)                # for video: (15, 299, 299), for image: (299, 299)
epochs = 100
max_sim_data_per_class = 950        # maximum number of synthetic scenarios per class to be used
max_real_data_per_class = 67        # maximum number of real scenarios per class to be used


SCENARIOS = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]
                                    # scenarios that are considered for training
PARAMS = {'dim': dim,
          'batch_size': 4,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True,
          'cnn_name': cnn_name,
          'classification': classification}


if classification == "video":
    model = my_model.build_video_model(SCENARIOS.__len__(), cnn_name, dropout)
elif classification == "image":
    model = my_model.build_image_model(SCENARIOS.__len__(), cnn_name, dropout)
else:
    model = False
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(1e-4), metrics=["accuracy"])
print(model.summary())


train_sim, val_sim, test_sim, label_dict = my_generator.get_data_and_labels(input_directory_sim, SCENARIOS,
                                                                            max_number=max_sim_data_per_class,
                                                                            train_share=0.70, val_share=0.90)
train_real, val_real, test_real, label_real = my_generator.get_data_and_labels(input_directory_real, SCENARIOS,
                                                                               max_number=max_real_data_per_class,
                                                                               train_share=0.50, val_share=0.75)

train_list = train_sim + train_real
val_list = val_sim + val_real
test_list = test_sim + test_real
label_dict.update(label_real)

random.shuffle(train_list)
random.shuffle(val_list)
random.shuffle(test_list)

print("training data (sim/real): " + str(len(train_sim)) + "/" + str(len(train_real)))
print("validation data (sim/real): " + str(len(val_sim)) + "/" + str(len(val_real)))
print("test data (sim/real): " + str(len(test_sim)) + "/" + str(len(test_real)))

train_generator = my_generator.DataGenerator(train_list, label_dict, **PARAMS)
val_generator = my_generator.DataGenerator(val_list, label_dict, **PARAMS)


""" Create callback for saving model if it improved """
file_path = output_directory + "/model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                             save_weights_only=False, mode='auto')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=20, verbose=1, mode='auto',
                                               baseline=None, restore_best_weights=False)

callbacks_list = [checkpoint, early_stopping]


history = model.fit_generator(generator=train_generator, validation_data=val_generator,
                              callbacks=callbacks_list, epochs=epochs)

print("Saving model...")
model.save(output_directory + "/model.h5")
np.save(output_directory + "/history.npy", history)


""" Saving setting of this model """
settings = {"scenarios": SCENARIOS, "params": PARAMS, "epochs": epochs, "label_dict": label_dict,
            "train_sim": train_sim, "val_sim": val_sim, "test_sim": test_sim,
            "train_real": train_real, "val_real": val_real, "test_real": test_real,
            "train_list": train_list, "val_list": val_list, "test_list": test_list,
            "model_compile": ["categorical_crossentropy", "keras.optimizers.Adam(1e-4)", "accuracy"]}
np.save(output_directory + "/settings.npy", settings)


""" Saving paths to test (real and sim) data """
np.save(output_directory + "/labels_test_data_sim.npy", my_generator.get_labels(test_sim, SCENARIOS))
np.save(output_directory + "/labels_test_data_real.npy", my_generator.get_labels(test_real, SCENARIOS))


""" Saving predictions of test (real and sim) data """
pred_sim = []
for path in test_sim:
    pred_sim.append(my_generator.get_prediction(model, path, cnn_name, classification))
pred_sim = np.squeeze(np.array(pred_sim))
np.save(output_directory + "/predictions_test_data_sim.npy", pred_sim)

pred_real = []
for path in test_real:
    pred_real.append(my_generator.get_prediction(model, path, cnn_name, classification))
pred_real = np.squeeze(np.array(pred_real))
np.save(output_directory + "/predictions_test_data_real.npy", pred_real)
