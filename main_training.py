import my_model
import my_generator
import keras


DIRECTORY = "training"
SCENARIOS = ["free_cruising", "following", "catching_up"]
PARAMS = {'dim': (15, 299, 299),
          'batch_size': 2,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}


def training():
    model = my_model.build_model_inceptionV3_LSTM(SCENARIOS.__len__())
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(3e-4), metrics=["accuracy"])
    print(model.summary())
    train_generator, validation_generator = my_generator.build_data_generators(DIRECTORY, SCENARIOS, PARAMS)
    history = model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=2)
    model.save(DIRECTORY + "/my_model_v3_lstm.h5")
    return model


model = training()
