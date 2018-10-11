import keras
from my_generator import DataGenerator
import preprocessing as pp


NO_OF_LABELS = 2
ROOT_DIRECTORY = "/Users/manuel/Dropbox/_data/"


# Parameters
params = {'dim': (5, 224, 224),
          'batch_size': 10,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# Datasets
directory = ROOT_DIRECTORY + "videos/"
data, labels = pp.get_data_and_labels(directory)

print(labels)
print(data['train'])
print(data['validation'])

# Generators
train_generator = DataGenerator(data['train'], labels, **params)
validation_generator = DataGenerator(data['validation'], labels, **params)


# create input layer
video = keras.Input(shape=(None, 150, 150, 3), name="video")

# load existing CNN with pre-trained wheights
cnn = keras.applications.InceptionV3(weights="imagenet", include_top=False, pooling="avg")
cnn.trainable = False
# print(cnn.summary())

# create LSTM and output layer
frame_features = keras.layers.TimeDistributed(cnn)(video)
video_vector = keras.layers.LSTM(256)(frame_features)
x = keras.layers.Dense(128, activation="relu")(video_vector)
predictions = keras.layers.Dense(NO_OF_LABELS, activation='softmax', name='predictions')(x)

# build final model
model = keras.Model(inputs=video, outputs=predictions)
model.compile(optimizer=keras.optimizers.Adam(3e-4), loss=keras.losses.categorical_crossentropy)

print(model.summary())

# Train model on dataset
model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=5)
