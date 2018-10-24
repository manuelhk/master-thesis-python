import keras


def build_model(no_of_labels):
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
    predictions = keras.layers.Dense(units=no_of_labels, activation='softmax', name='predictions')(x)

    # build final model
    model = keras.Model(inputs=video, outputs=predictions)
    return model
