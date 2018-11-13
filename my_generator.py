import glob
import random
import os
import keras
import numpy as np


""" 
    Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 
            A detailed example of how to use data generators with Keras
"""


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """
    def __init__(self, list_IDs, labels, batch_size=20, dim=(299, 299), n_channels=3,
                 n_classes=3, shuffle=True):
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
        """ Generates data containing batch_size samples """
        # Initialization
        data = np.empty((self.batch_size, *self.dim, self.n_channels))
        labels = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            a = np.load(ID)
            data[i, ] = keras.applications.inception_v3.preprocess_input(a)
            labels[i] = self.labels[ID]
        return data, keras.utils.to_categorical(labels, num_classes=self.n_classes)


def get_data_and_labels(directory, scenarios, max_number=950):
    paths_train = []
    paths_val = []
    paths_test = []
    for label in scenarios:
        p = glob.glob(directory + os.sep + str(label) + os.sep + "*.npy")
        random.shuffle(p)
        # print(label + ": " + str(p.__len__()))
        for i in range(max_number):
            if i < int(max_number*0.85):
                paths_train.append(p[i])
            elif i < int(max_number*0.95):
                paths_val.append(p[i])
            else:
                paths_test.append(p[i])
    paths_all = paths_test + paths_val + paths_train
    labels_dict = dict()
    for path in paths_all:
        for label in scenarios:
            if label in path:
                labels_dict.update({path: scenarios.index(label)})
    random.shuffle(paths_train)
    random.shuffle(paths_val)
    random.shuffle(paths_test)
    return paths_train, paths_val, paths_test, labels_dict


def get_labels(paths_to_data, scenarios):
    l = []
    for path in paths_to_data:
        for label in scenarios:
            if label in path:
                l.append(scenarios.index(label))
    labels = np.array(l)
    return labels


def get_data(paths_to_data):
    d = []
    for path in paths_to_data:
        d.append(keras.applications.inception_v3.preprocess_input(np.load(path)))
    labels = np.array(d)
    return labels


"""
def build_data_generators(directory, scenarios, params):
    data, labels = get_data_and_labels_old(directory, scenarios)
    train_generator = DataGenerator(data['train'], labels, **params)
    validation_generator = DataGenerator(data['validation'], labels, **params)
    return train_generator, validation_generator


def get_data_and_labels_old(directory, scenarios):
    paths = glob.glob(directory + "/*/*.npy")
    # paths.sort()
    labels_dict = dict()
    for path in paths:
        for label in scenarios:
            if label in path:
                labels_dict.update({path: scenarios.index(label)})
    data_dict = split_into_training_and_validation(paths)
    return data_dict, labels_dict


def split_into_training_and_validation(list_of_paths, share_training_data=0.8):
    random.shuffle(list_of_paths)
    train_list = list_of_paths[:np.math.floor(len(list_of_paths)*share_training_data)]
    eval_list = list_of_paths[np.math.floor(len(list_of_paths)*share_training_data):]
    data_dict = {"train": train_list, "validation": eval_list}
    return data_dict
"""
