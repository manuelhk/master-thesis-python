import my_model
import my_generator


DIRECTORY = "test_data"


model = my_model.build_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
train_generator, validation_generator = my_generator.build_data_generators(DIRECTORY)
model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=5)
