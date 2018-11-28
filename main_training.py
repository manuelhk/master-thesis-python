import my_model
import my_generator
import keras
import random
import numpy as np


input_directory_sim = "input"
input_directory_real = "input/real"
output_directory = "output"

classification = "video"        # video or image
cnn_name = "inception_v3"       # inception_v3 or xception
dropout = True
dim = (15, 299, 299)            # for video: (15, 299, 299), for image: (299, 299)
epochs = 5
max_sim_data_per_class = 950
max_real_data_per_class = 67


SCENARIOS = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]
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
                                                                            train_share=0.85, val_share=0.95)
train_real, val_real, test_real, label_real = my_generator.get_data_and_labels(input_directory_real, SCENARIOS,
                                                                               max_number=max_real_data_per_class,
                                                                               train_share=0.65, val_share=0.75)

train_list = train_sim + train_real
val_list = val_sim + val_real
test_list = test_sim + test_real
label_dict.update(label_real)

random.shuffle(train_list)
random.shuffle(val_list)
random.shuffle(test_list)

print(str(train_list.__len__()) + " objects in training data")
print(str(val_list.__len__()) + " objects in validation data")
print(str(test_list.__len__()) + " objects in test data")
print(str(test_real.__len__()) + " objects in test data real")

train_generator = my_generator.DataGenerator(train_list, label_dict, **PARAMS)
val_generator = my_generator.DataGenerator(val_list, label_dict, **PARAMS)


""" Create callback for saving model if it improved """
# file_path = output_directory + "/model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True,
#                                              save_weights_only=False, mode='auto')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=5, verbose=1, mode='auto',
                                               baseline=None, restore_best_weights=True)

callbacks_list = [early_stopping]


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


np.save(output_directory + "/labels_test_data.npy", my_generator.get_labels(test_list, SCENARIOS))
test_list_generator = my_generator.DataGenerator(test_list, label_dict, **PARAMS)
np.save(output_directory + "/predictions_test_data.npy", model.predict_generator(test_list_generator))

np.save(output_directory + "/labels_test_data_sim.npy", my_generator.get_labels(test_sim, SCENARIOS))
test_sim_generator = my_generator.DataGenerator(test_sim, label_dict, **PARAMS)
np.save(output_directory + "/predictions_test_data_sim.npy", model.predict_generator(test_sim_generator))

np.save(output_directory + "/labels_test_data_real.npy", my_generator.get_labels(test_real, SCENARIOS))
test_real_generator = my_generator.DataGenerator(test_real, label_dict, **PARAMS)
np.save(output_directory + "/predictions_test_data_real.npy", model.predict_generator(test_real_generator))
