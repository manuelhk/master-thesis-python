import keras
import preprocessing
import numpy as np


""" 
    Article: A detailed example of how to use data generators with Keras
    Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 
"""


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """
    def __init__(self, list_IDs, labels, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=10, shuffle=True):
        """ Initialization """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        data, labels = self.__data_generation(list_IDs_temp)

        return data, labels

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """ Generates data containing batch_size samples """  # X : (n_samples, *dim, n_channels)
        # Initialization
        data = np.empty((self.batch_size, *self.dim, self.n_channels))
        labels = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data[i, ] = np.load(ID)
            # Store class
            labels[i] = self.labels[ID]
        return data, keras.utils.to_categorical(labels, num_classes=self.n_classes)


def build_data_generators(directory):
    params = {'dim': (25, 150, 150),
              'batch_size': 5,
              'n_classes': 2,
              'n_channels': 3,
              'shuffle': True}
    data, labels = preprocessing.get_data_and_labels(directory)
    train_generator = DataGenerator(data['train'], labels, **params)
    validation_generator = DataGenerator(data['validation'], labels, **params)
    return train_generator, validation_generator
