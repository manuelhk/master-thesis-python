import glob
import random
import keras
import numpy as np


""" 
    Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly 
            A detailed example of how to use data generators with Keras
"""


class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """
    def __init__(self, list_IDs, labels, batch_size=16, dim=(299, 299), n_channels=3,
                 n_classes=5, shuffle=True, cnn_name="inception_v3", classification="video"):
        """ Initialization """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.cnn_name = cnn_name
        self.classification = classification
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
        if self.classification == "video":
            # Initialization
            data = np.empty((self.batch_size, *self.dim, self.n_channels))
            labels = np.empty(self.batch_size, dtype=int)
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                a = np.load(ID)
                if self.cnn_name == "inception_v3":
                    data[i, ] = keras.applications.inception_v3.preprocess_input(a)
                elif self.cnn_name == "xception":
                    data[i, ] = keras.applications.xception.preprocess_input(a)
                else:
                    print("ERROR: cnn_name not valid")
                labels[i] = self.labels[ID]
            return data, keras.utils.to_categorical(labels, num_classes=self.n_classes)
        elif self.classification == "image":
            # Initialization
            data = np.empty((self.batch_size*15, *self.dim, self.n_channels))
            labels = np.empty(self.batch_size*15, dtype=int)
            # Generate data
            count = 0
            for ID in list_IDs_temp:
                a = np.load(ID)
                if self.cnn_name == "inception_v3":
                    for j in range(15):
                        data[count, ] = keras.applications.inception_v3.preprocess_input(a[j])
                        labels[count] = self.labels[ID]
                        count += 1
                elif self.cnn_name == "xception":
                    for j in range(15):
                        data[count, ] = keras.applications.xception.preprocess_input(a[j])
                        labels[count] = self.labels[ID]
                        count += 1
                else:
                    print("ERROR: cnn_name not valid")
            return data, keras.utils.to_categorical(labels, num_classes=self.n_classes)
        else:
            print("ERROR: classification not valid")
        pass


def get_data_and_labels(directory, scenarios, max_number=950, train_share=0.85, val_share=0.95):
    paths_train = []
    paths_val = []
    paths_test = []
    for label in scenarios:
        p = glob.glob(directory + "/" + str(label) + "/" + "*.npy")
        random.shuffle(p)
        # print(label + ": " + str(p.__len__()))
        for i in range(max_number):
            if i < int(max_number*train_share):
                paths_train.append(p[i])
            elif i < int(max_number*val_share):
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
    label_list = []
    for path in paths_to_data:
        for label in scenarios:
            if label in path:
                label_list.append(scenarios.index(label))
    labels = np.array(label_list)
    return labels


"""
def get_data(paths_to_data, cnn_name="inception_v3"):
    d = []
    for path in paths_to_data:
        if cnn_name == "inception_v3":
            d.append(keras.applications.inception_v3.preprocess_input(np.load(path)))
        elif cnn_name == "xception":
            d.append(keras.applications.xception.preprocess_input(np.load(path)))
        else:
            print("ERROR: cnn_name not defined")
    labels = np.array(d)
    return labels
"""
