import my_model
import my_generator
import keras


DIRECTORY = "training"
SCENARIOS = ["free_cruising", "following", "catching_up"]
PARAMS = {'dim': (15, 299, 299),
          'batch_size': 2,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}


train_generator, validation_generator = my_generator.build_data_generators(DIRECTORY, SCENARIOS, PARAMS)


