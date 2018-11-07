import my_generator
import math


DIRECTORY = "training"
SCENARIOS = ["lane_change_left", "lane_change_right"]
PARAMS = {'dim': (15, 299, 299),
          'batch_size': 2,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}


data_paths, label_dict = my_generator.get_data_and_labels(DIRECTORY, SCENARIOS)

train_list = data_paths[:math.floor(len(data_paths)*0.8)]
eval_list = data_paths[math.floor(len(data_paths)*0.15):math.floor(len(data_paths)*0.95)]
test_list =data_paths[:math.floor(len(data_paths)*0.05)]

train_generator = my_generator.DataGenerator(train_list, label_dict, **PARAMS)
validation_generator = my_generator.DataGenerator(eval_list, label_dict, **PARAMS)
test_generator = my_generator.DataGenerator(test_list, label_dict, **PARAMS)
