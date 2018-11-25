import keras


def build_video_model(no_of_labels, cnn_name="inception_v3", dropout=True):
    video = keras.Input(shape=(15, 299, 299, 3), name="video")
    if cnn_name == "inception_v3":
        cnn = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")
        print("inception_v3")
    elif cnn_name == "xception":
        cnn = keras.applications.xception.Xception(weights="imagenet", include_top=False, pooling="avg")
        print("xception")
    cnn.trainable = False
    x = keras.layers.TimeDistributed(cnn)(video)
    x = keras.layers.LSTM(256)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    if dropout:
        x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=video, outputs=output)
    return model
