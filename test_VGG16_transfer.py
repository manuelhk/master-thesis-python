import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import read_files as rf
import glob


# sources:
# https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/
# https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
# https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/
# https://www.youtube.com/watch?v=dfQ8lZ9dTjs


# create input layer
image = tf.keras.Input(shape=(224, 224, 3), name="video")

print("Load model...")
model = tf.keras.applications.vgg16.VGG16(weights="imagenet",
                                          include_top=False,
                                          input_shape=(224, 224, 3))
model.trainable = False
# print(model.summary())
# plot_model(model, to_file='vgg.png')

# adding custom layers
x = model(image)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, activation="relu")(x)
predictions = tf.keras.layers.Dense(2, activation="softmax")(x)

# creating the final model
model_final = tf.keras.models.Model(inputs=image, outputs=predictions)

# compile the model
model_final.compile(optimizer=tf.train.AdamOptimizer(3e-4),
                    loss=keras.losses.categorical_crossentropy,
                    metrics=[keras.metrics.categorical_accuracy])

print(model_final.summary())


estimator = tf.keras.estimator.model_to_estimator(model_final, model_dir="nn/log")

train_files = glob.glob("nn/train/*/*.jpg")
train_labels = [0 if "left" in file_path else 1 for file_path in train_files]
eval_files = glob.glob("nn/validation/*/*.jpg")
eval_labels = [0 if "left" in file_path else 1 for file_path in eval_files]


next_batch = rf.input_fn(train_files, train_labels)

sess = tf.Session()
video, label = sess.run(next_batch)
print("videos:", video["video"].shape)
print("labels:", label)

plt.imshow(video["video"][0, :, :, :])
plt.show()


train_input = lambda: rf.input_fn(train_files, train_labels)
train_spec = tf.estimator.TrainSpec(train_input, max_steps=10000)

eval_input = lambda: rf.input_fn(eval_files, eval_labels)
eval_spec = tf.estimator.EvalSpec(eval_input, steps=100)

print(train_spec)
print(eval_spec)


tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

"""
"""
"""

train_files = tf.constant(train_files)
train_labels = tf.constant(train_labels)
eval_files = tf.constant(eval_files)
eval_labels = tf.constant(eval_labels)

"""
'''
train_dir = "nn/train"
validate_dir = "nn/validation"
img_height = 224
img_width = 224
batch_size = 10
nb_train = 750
nb_val = 100
epochs = 5


# Initiate the train and test generators with data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validate_dir,
    target_size=(img_height, img_width),
    class_mode="categorical")


# Save the model according to the conditions
checkpoint = ModelCheckpoint(
    "vgg16_1.h5",
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1)
early = EarlyStopping(
    monitor='val_acc',
    min_delta=0,
    patience=10,
    verbose=1,
    mode='auto')

# Train the model
model_final.fit_generator(
    train_generator,
    samples_per_epoch=nb_train,
    epochs=epochs,
    validation_data=validation_generator)
    nb_val_samples=nb_val,
    callbacks=[checkpoint, early])


model_final.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)
'''
