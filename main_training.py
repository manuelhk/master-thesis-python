import my_model
import my_generator


DIRECTORY = "output"
SCENARIOS = ["free_cruising", "following", "overtaking"]
PARAMS = {'dim': (15, 150, 150),
          'batch_size': 5,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}


def training():
    print("------------------ Training... ------------------")
    model = my_model.build_model(SCENARIOS.__len__())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    train_generator, validation_generator = my_generator.build_data_generators(DIRECTORY, SCENARIOS, PARAMS)
    model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=5)
    print("-------------------------------------------------")
    pass


training()
