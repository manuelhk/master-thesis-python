import keras


################################################################################
################################################################################

# Methods in this script design neural networks based on given parameters.
# Generally it is differentiated between a model that classifies single images
# (build_image_model) and a model that classifies a sequence of images (build_video_model)

################################################################################
################################################################################


def build_video_model(no_of_labels, cnn_name="inception_v3", dropout=True):
    """
    This method designs a model that classifies a sequence of images

    :param no_of_labels: number of classes to be classified
    :param cnn_name: name of the cnn to be used (Inception-V3 or Xception)
    :param dropout: indicated whether or not dropout is used in the second to last layer
    :return: keras model
    """
    video = keras.Input(shape=(15, 299, 299, 3), name="video")
    if cnn_name == "inception_v3":
        cnn = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg")
        print("base model: inception_v3")
    elif cnn_name == "xception":
        cnn = keras.applications.xception.Xception(weights="imagenet", include_top=False, pooling="avg")
        print("base model: xception")
    cnn.trainable = False
    x = keras.layers.TimeDistributed(cnn)(video)
    x = keras.layers.LSTM(256)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    if dropout:
        x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=video, outputs=output)
    return model


def build_image_model(no_of_labels, cnn_name="inception_v3", dropout=True):
    """
    This method designs a model that classifies single images

    :param no_of_labels: number of classes to be classified
    :param cnn_name: name of the cnn to be used (Inception-V3 or Xception)
    :param dropout: indicated whether or not dropout is used in the second to last layer
    :return: keras model
    """
    if cnn_name == "inception_v3":
        cnn = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg",
                                                          input_shape=(299, 299, 3))
        print("base model: inception_v3")
    elif cnn_name == "xception":
        cnn = keras.applications.xception.Xception(weights="imagenet", include_top=False, pooling="avg",
                                                   input_shape=(299, 299, 3))
        print("base model: xception")
    cnn.trainable = False
    x = cnn.output
    # x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    if dropout:
        x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(units=no_of_labels, activation="softmax", name='predictions')(x)
    model = keras.Model(inputs=cnn.input, outputs=output)
    return model
