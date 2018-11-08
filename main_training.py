import my_model
import my_generator
import keras
import math


DIRECTORY = "input"
SCENARIOS = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]
PARAMS = {'dim': (15, 299, 299),
          'batch_size': 4,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}


model = my_model.build_model_inceptionV3_LSTM(SCENARIOS.__len__())
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(3e-4), metrics=["accuracy"])
print(model.summary())

data_paths, label_dict = my_generator.get_data_and_labels(DIRECTORY, SCENARIOS, max_number=950)
train_list = data_paths[:math.floor(len(data_paths)*0.9)]
val_list = data_paths[math.floor(len(data_paths)*0.9):math.floor(len(data_paths)*0.95)]
test_list = data_paths[math.floor(len(data_paths)*0.95):]

train_generator = my_generator.DataGenerator(train_list, label_dict, **PARAMS)
val_generator = my_generator.DataGenerator(val_list, label_dict, **PARAMS)
test_generator = my_generator.DataGenerator(test_list, label_dict, **PARAMS)

history = model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=3)
model.save("output/1108_v3_lstm_fr_fo_ca_lcl_lcr_950_10.h5")
