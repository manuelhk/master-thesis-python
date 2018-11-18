import my_model
import my_generator
import keras
import random
import numpy as np


DIRECTORY_SIM = "input"
DIRECTORY_REAL = "input/real"
SCENARIOS = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]
PARAMS = {'dim': (15, 299, 299),
          'batch_size': 4,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}
EPOCHS = 10


model = my_model.build_model_inceptionV3_LSTM_dropout(SCENARIOS.__len__())
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(1e-4), metrics=["accuracy"])
print(model.summary())

train_sim, val_sim, test_sim, label_dict = my_generator.get_data_and_labels(DIRECTORY_SIM, SCENARIOS, max_number=950,
                                                                            train_share=0.85, val_share=0.95)
train_real, val_real, test_real, label_real = my_generator.get_data_and_labels(DIRECTORY_REAL, SCENARIOS, max_number=67,
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

train_generator = my_generator.DataGenerator(train_list, label_dict, **PARAMS)
val_generator = my_generator.DataGenerator(val_list, label_dict, **PARAMS)


""" Create callback for saving model if it improved """
file_path = "output/model-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                             save_weights_only=False, mode='auto')
callbacks_list = [checkpoint]


""" Saving setting of this model """
settings = {"scenarios": SCENARIOS, "params": PARAMS, "epochs": EPOCHS, "label_dict": label_dict,
            "train_sim": train_sim, "val_sim": val_sim, "test_sim": test_sim,
            "train_real": train_real, "val_real": val_real, "test_real": test_real,
            "train_list": train_list, "val_list": val_list, "test_list": test_list,
            "model_compile": ["categorical_crossentropy", "keras.optimizers.Adam(1e-4)", "accuracy"]}
np.save("output/settings.npy", settings)


history = model.fit_generator(generator=train_generator, validation_data=val_generator,
                              callbacks=callbacks_list, epochs=EPOCHS)

print("Saving model...")
model.save("output/model.h5")
np.save("output/history.npy", history)


np.save("output/labels_test_data.npy", my_generator.get_labels(test_list, SCENARIOS))
np.save("output/predictions_test_data.npy", model.predict(my_generator.get_data(test_list)))

np.save("output/labels_test_data_sim.npy", my_generator.get_labels(test_sim, SCENARIOS))
np.save("output/predictions_test_data_sim.npy", model.predict(my_generator.get_data(test_sim)))

np.save("output/labels_test_data_real.npy", my_generator.get_labels(test_real, SCENARIOS))
np.save("output/predictions_test_data_real.npy", model.predict(my_generator.get_data(test_real)))
