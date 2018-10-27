import keras


def build_model_inceptionV3_LSTM(no_of_labels):
    video = keras.Input(shape=(15, 150, 150, 3), name="video")
    cnn = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    x = keras.layers.TimeDistributed(cnn)(video)
    x = keras.layers.LSTM(256, return_sequences=True)(x)
    x = keras.layers.LSTM(256, return_sequences=True)(x)
    x = keras.layers.LSTM(256)(x)

    # x = keras.layers.Flatten()(x)
    # x = keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dense(1024, activation="relu")(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=video, outputs=output)
    return model


def build_model_VGG16_LSTM(no_of_labels):
    video = keras.Input(shape=(15, 150, 150, 3), name="video")
    cnn = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', pooling="avg")
    frame_features = keras.layers.TimeDistributed(cnn)(video)
    video_vector = keras.layers.LSTM(128)(frame_features)
    x = keras.layers.Dense(256, activation="relu")(video_vector)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=video, outputs=output)
    return model


def build_model_VGG16(no_of_labels):
    image = keras.Input(shape=(150, 150, 3), name="image")
    cnn = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
    x = cnn(image)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation="relu")(x)
    output = keras.layers.Dense(no_of_labels, activation="softmax")(x)
    model = keras.Model(inputs=image, outputs=output)
    return model


def build_my_model(no_of_labels):
    model = keras.Sequential()
    model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3)), input_shape=(15, 150, 150, 3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(no_of_labels))
    return model
