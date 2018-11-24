import keras


def build_model_inceptionV3_LSTM(no_of_labels):
    video = keras.Input(shape=(15, 299, 299, 3), name="video")
    cnn = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False
    x = keras.layers.TimeDistributed(cnn)(video)
    x = keras.layers.LSTM(256)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=video, outputs=output)
    return model


def build_model_inceptionV3_LSTM_dropout(no_of_labels):
    video = keras.Input(shape=(15, 299, 299, 3), name="video")
    cnn = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False
    x = keras.layers.TimeDistributed(cnn)(video)
    x = keras.layers.LSTM(256)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=video, outputs=output)
    return model


def build_model_inceptionV3(no_of_labels):
    image = keras.Input(shape=(299, 299, 3), name="image")
    cnn = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")(image)
    cnn.trainable = False
    x = keras.layers.Dense(128, activation="relu")(cnn)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=image, outputs=output)
    return model


def build_model_inceptionV3_dropout(no_of_labels):
    image = keras.Input(shape=(299, 299, 3), name="image")
    cnn = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")(image)
    cnn.trainable = False
    x = keras.layers.Dense(128, activation="relu")(cnn)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=image, outputs=output)
    return model


def build_model_vgg16_LSTM(no_of_labels):
    video = keras.Input(shape=(15, 299, 299, 3), name="video")
    cnn = keras.applications.VGG16(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False
    x = keras.layers.TimeDistributed(cnn)(video)
    x = keras.layers.LSTM(256)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=video, outputs=output)
    return model


def build_model_vgg16_LSTM_dropout(no_of_labels):
    video = keras.Input(shape=(15, 299, 299, 3), name="video")
    cnn = keras.applications.VGG16(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False
    x = keras.layers.TimeDistributed(cnn)(video)
    x = keras.layers.LSTM(256)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=video, outputs=output)
    return model



def build_model_vgg16(no_of_labels):
    image = keras.Input(shape=(299, 299, 3), name="image")
    cnn = keras.applications.VGG16(weights="imagenet", include_top=False, pooling="avg")(image)
    cnn.trainable = False
    x = keras.layers.Dense(128, activation="relu")(cnn)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=image, outputs=output)
    return model


def build_model_vgg16_dropout(no_of_labels):
    image = keras.Input(shape=(299, 299, 3), name="image")
    cnn = keras.applications.VGG16(weights="imagenet", include_top=False, pooling="avg")(image)
    cnn.trainable = False
    x = keras.layers.Dense(128, activation="relu")(cnn)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=image, outputs=output)
    return model
