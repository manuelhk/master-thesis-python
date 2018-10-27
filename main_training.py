import my_model
import my_generator
import keras
import time


DIRECTORY = "test"
SCENARIOS = ["free_cruising", "following", "catching_up"]
PARAMS = {'dim': (15, 150, 150),
          'batch_size': 20,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}


def training():
    model = my_model.build_model_inceptionV3_LSTM(SCENARIOS.__len__())
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.1), metrics=["accuracy"])
    print(model.summary())
    train_generator, validation_generator = my_generator.build_data_generators(DIRECTORY, SCENARIOS, PARAMS)
    model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=2)
    model.save("test/my_model_v3_lstm.h5")
    return model


model = training()
