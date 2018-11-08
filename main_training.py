import my_model
import my_generator
import keras
import math
import numpy as np


DIRECTORY = "input"
SCENARIOS = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]
PARAMS = {'dim': (15, 299, 299),
          'batch_size': 4,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}
EPOCHS = 20


model = my_model.build_model_inceptionV3_LSTM(SCENARIOS.__len__())
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(3e-4), metrics=["accuracy"])
print(model.summary())

data_paths, label_dict = my_generator.get_data_and_labels(DIRECTORY, SCENARIOS, max_number=950)
train_list = data_paths[:math.floor(len(data_paths)*0.9)]
val_list = data_paths[math.floor(len(data_paths)*0.9):math.floor(len(data_paths)*0.95)]
test_list = data_paths[math.floor(len(data_paths)*0.95):]

print(str(train_list.__len__()) + " objects in training data")
print(str(val_list.__len__()) + " objects in validation data")
print(str(test_list.__len__()) + " objects in test data")

train_generator = my_generator.DataGenerator(train_list, label_dict, **PARAMS)
val_generator = my_generator.DataGenerator(val_list, label_dict, **PARAMS)

history = model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=EPOCHS)

model.save("output/model.h5")
np.save("output/history.npy", history)
np.save("output/labels_test_data.npy", my_generator.get_labels(test_list, SCENARIOS))
np.save("output/predictions_test_data.npy", model.predict(my_generator.get_data(test_list)))
