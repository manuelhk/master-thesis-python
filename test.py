import my_model


# model = keras.applications.VGG16()
# print(model.summary())
# keras.utils.plot_model(model, to_file='model.png')

model = my_model.build_model_vgg16_LSTM_dropout(5)
print(model.summary())
