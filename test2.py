from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.layers import TimeDistributed
import my_generator


DIRECTORY = "test"
SCENARIOS = ["free_cruising", "following", "catching_up"]
PARAMS = {'dim': (15, 150, 150),
          'batch_size': 8,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}

# First, let's define a vision model using a Sequential model
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Now let's get a tensor with the output of our vision model:
image_input = Input(shape=(150, 150, 3))
encoded_image = vision_model(image_input)

video_input = Input(shape=(15, 150, 150, 3))
# This is our video encoded via the previously trained vision_model (weights are reused)
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

output = Dense(3, activation='softmax')(encoded_video)
video_model = Model(inputs=video_input, outputs=output)

video_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
print(video_model.summary())
train_generator, validation_generator = my_generator.build_data_generators(DIRECTORY, SCENARIOS, PARAMS)
video_model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=3)
video_model.save("test/my_video_model5")
