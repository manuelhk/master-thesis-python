import keras
from my_generator import DataGenerator
import preprocessing as pp

NO_OF_LABELS = 2
ROOT_DIRECTORY = "data/"

# Parameters
params = {'dim': (224, 224),
          'batch_size': 10,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# Datasets
directory = ROOT_DIRECTORY + "nn/data/"
data, labels = pp.get_data_and_labels(directory)

print(labels)
print(data['train'])
print(data['validation'])

# Generators
train_generator = DataGenerator(data['train'], labels, **params)
validation_generator = DataGenerator(data['validation'], labels, **params)

print(train_generator)
print(validation_generator)

# create input layer
image = keras.Input(shape=(224, 224, 3), name="video")

print("Load model...")
model = keras.applications.vgg16.VGG16(weights="imagenet",
                                       include_top=False,
                                       input_shape=(224, 224, 3))
model.trainable = False
# print(model.summary())
# plot_model(model, to_file='vgg.png')

# adding custom layers
x = model(image)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024, activation="relu")(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(1024, activation="relu")(x)
predictions = keras.layers.Dense(2, activation="softmax")(x)

# creating the final model
model_final = keras.models.Model(inputs=image, outputs=predictions)

# compile the model
model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model_final.summary())


# Train model on dataset
model_final.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=5)


