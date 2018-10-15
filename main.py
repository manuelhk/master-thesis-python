import labeling
# import time
import my_model
import my_generator


directory = "test_data/"


# print("Labeling data...")
# start = time.clock()
data, metadata = labeling.get_data(dir + "data/0810/hc/data.dat")
scenes_labeled, scenarios_labeled = labeling.label_scenarios(data, metadata)
labeling.save_video(dir + "data/0810/hc/frames/", dir + "data/0810/hc/video.avi", scenes_labeled, scenarios_labeled)
# diff = time.clock() - start
# print("Labeling data took " + str(round(diff, 2)) + " seconds")


""" Build, compile and train model """
# model = my_model.build_model()
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# train_generator, validation_generator = my_generator.build_data_generators(directory)
# model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=5)
