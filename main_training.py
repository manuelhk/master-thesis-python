import my_model
import my_generator


DIRECTORY = "test_data"
NO_OF_LABELS = 2
PARAMS = {'dim': (25, 150, 150),
          'batch_size': 5,
          'n_classes': NO_OF_LABELS,
          'n_channels': 3,
          'shuffle': True}


model = my_model.build_model(NO_OF_LABELS)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
train_generator, validation_generator = my_generator.build_data_generators(DIRECTORY, PARAMS)
model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=5)
