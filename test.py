import my_generator


DIRECTORY = "output"
SCENARIOS = ["free_cruising", "following", "overtaking"]
PARAMS = {'dim': (15, 150, 150),
          'batch_size': 5,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True}