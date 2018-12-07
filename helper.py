import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import itertools
import os


################################################################################
################################################################################

# This script contains various methods for preparing training data or
# visualizing results of trained neural nets and test data

################################################################################
################################################################################


def video_to_jpges_and_npys(video_path, output_path):
    """
        This method converts a video into numpy arrays with 15 frames each
        it is used for preparing real data from a camera in a car into training input for the neural net
        reshaping should be adapted to input shape -> see below
    """
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success = True
    a = []
    while success:
        success, image = vidcap.read()
        if not success:
            break
        cv2.imwrite(output_path + "frame%d.jpg" % count, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:, 106:746, :]                    # reshaping should be adapted to input shape
        image = cv2.resize(image, (299, 299))
        count += 1
        array = np.array(image)
        a.append(array)
        if count % 15 == 0:
            np.save(output_path + str(count) + ".npy", np.array(a))
            a = []
    pass


def show_npy(path, number_of_images=3):
    """
        This method is used to visualize a numpy array containing 15 images
        the number of frames to be shown can be specified
    """
    array = np.load(path)
    index = 1
    for i in range(0, 15, int(15/number_of_images)):
        plt.imshow(array[i])
        plt.xlabel(path + " (" + str(i) + ")")
        plt.show()
        index += 1
    pass


def show_training_history(history_path):
    """ Plotting accuracy and loss of training and validation data from training history object (Keras)"""
    print("Load history...")
    history_np = np.load(history_path)
    epochs = len(history_np.item().history.get("acc"))
    if epochs < 20:
        step = 1
    elif epochs < 30:
        step = 2
    elif epochs < 40:
        step = 3
    else:
        step = 5
    # plot the model's accuracy
    fig = plt.figure()
    plt.plot(history_np.item().history.get("acc"))
    plt.plot(history_np.item().history.get("val_acc"))
    # plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.xticks(np.arange(start=step, step=step, stop=epochs+1) - 1,
               np.arange(start=step, step=step, stop=epochs+1, dtype="int"))
    plt.ylim(bottom=0.8, top=1)
    plt.xlim(left=0, right=epochs-1)
    plt.show()
    fig.savefig("output/accuracy.png")
    # plot the model's loss
    fig = plt.figure()
    plt.plot(history_np.item().history.get("loss"))
    plt.plot(history_np.item().history.get("val_loss"))
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xticks(np.arange(start=step, step=step, stop=epochs + 1) - 1,
               np.arange(start=step, step=step, stop=epochs + 1, dtype="int"))
    plt.ylim(bottom=0, top=1)
    plt.xlim(left=0, right=epochs-1)
    plt.legend(["Train", "Validation"])
    plt.show()
    fig.savefig("output/loss.png")
    pass


def show_confusion_matrix(y_true, y_pred, label_names, normalize=False, title="Confusion matrix"):
    """ Plots a confusion matrix """
    cm = confusion_matrix(y_true, y_pred)
    acc = np.round((cm[0, 0] + cm[1, 1] + cm[2, 2] + cm[3, 3] + cm[4, 4]) / np.sum(cm), 4)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig.savefig("output/" + title + " acc " + str(acc) + ".png")
    pass


def show_results(model, normalize=False):
    """
    Shows both the confusion matrix and the loss and accuracy functions
    This function is specified for the authors directory and need to be adapted

    :param model: name of the folder that contains history object, settings object, scenarios array,
                  labels_test_data_sim array, predictions_test_data_sim array, labels_test_data_real array and
                  predictions_test_data_real array
    :param normalize: specifies if confusion matrix should be normalized
    :return: saves loss, accuracy and cofusion matrix in project directory
    """
    show_training_history("/Users/manuel/Dropbox/_data/_models/" + model + "/history.npy")
    settings = np.load("/Users/manuel/Dropbox/_data/_models/" + model + "/settings.npy")
    settings = settings.item()
    label_names = settings["scenarios"]
    # Confusion matrix synthetic data
    y_true = np.load("/Users/manuel/Dropbox/_data/_models/" + model + "/labels_test_data_sim.npy")
    y_pred = np.argmax(np.load("/Users/manuel/Dropbox/_data/_models/" + model + "/predictions_test_data_sim.npy"), 1)
    show_confusion_matrix(y_true, y_pred, label_names, normalize=normalize, title="Confusion matrix synthetic data")
    # Confusion matrix real data
    y_true = np.load("/Users/manuel/Dropbox/_data/_models/" + model + "/labels_test_data_real.npy")
    y_pred = np.argmax(np.load("/Users/manuel/Dropbox/_data/_models/" + model + "/predictions_test_data_real.npy"), 1)
    show_confusion_matrix(y_true, y_pred, label_names, normalize=normalize, title="Confusion matrix real data")
    pass
