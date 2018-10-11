import keras
import tensorflow as tf
from keras import layers
from keras.applications import InceptionV3


NO_OF_LABELS = 3
TRAIN_FILES = ""
EVAL_FILES = ""


def input_fn(filenames,
             epochs=50,
             batch_size=10):
    # parse files and create dataset
    dataset = tf.data.Dataset.from_tensor_slices(...)

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    video, labels = iterator.get_next()
    return {"video": video}, labels


# create input layer
video = keras.Input(shape=(None, 150, 150, 3), name="video")

# load existing CNN with pre-trained wheights
cnn = InceptionV3(weights="imagenet",
                  include_top=False,
                  pooling="avg")
cnn.trainable = False
print(cnn.summary())

# create LSTM and output layer
frame_features = layers.TimeDistributed(cnn)(video)
video_vector = layers.LSTM(256)(frame_features)
x = layers.Dense(128, activation=tf.nn.relu)(video_vector)
predictions = layers.Dense(NO_OF_LABELS,
                           activation='softmax',
                           name='predictions')(x)

# build final model
model = keras.Model(inputs=video, outputs=predictions)
model.compile(optimizer=tf.train.AdamOptimizer(3e-4),
              loss=keras.losses.categorical_crossentropy)

# build estimator model
# config = {output_dir: "...", ...}
# estimator = keras.estimator.model_to_estimator(keras_model=model)



