import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import itertools
import os


def video_to_jpges_and_npys(video_path, output_path):
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
        image = image[:, 106:746, :]
        image = cv2.resize(image, (299, 299))
        count += 1
        array = np.array(image)
        a.append(array)
        if count % 15 == 0:
            np.save(output_path + str(count) + ".npy", np.array(a))
            a = []
    pass


def video_to_jpges(video_path, output_path, folder_start=0):
    vid_cap = cv2.VideoCapture(video_path)
    count = 0
    folder = folder_start
    os.mkdir(output_path + str(folder))
    success = True
    while success:
        success, image = vid_cap.read()
        if not success:
            break
        image = image[:, 239:1679, :]
        cv2.imwrite(output_path + str(folder) + "/frame%d.jpg" % count, image)
        count += 1
        if count % 15 == 0:
            folder += 1
            count = 0
            os.mkdir(output_path + str(folder))
        if folder % 100 == 0 and count == 0:
            print("folder: " + str(folder))
    pass


def show_npy(path, number_of_images=3):
    array = np.load(path)
    index = 1
    for i in range(0, 15, int(15/number_of_images)):
        # plt.plot(1, number_of_images, index)
        plt.imshow(array[i])
        plt.xlabel(path + " (" + str(i) + ")")
        plt.show()
        index += 1
    # plt.suptitle(title)
    # plt.show()
    pass


def show_training_history(history_path):
    """ Plotting accuracy and loss of training and validation data from training history object """
    print("Load history...")
    history_np = np.load(history_path)
    epochs = len(history_np.item().history.get("acc"))
    step = epochs/10
    # plot the model's accuracy
    fig = plt.figure()
    plt.plot(history_np.item().history.get("acc"))
    plt.plot(history_np.item().history.get("val_acc"))
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.xticks(np.arange(start=step, step=step, stop=epochs+1) - 1,
               np.arange(start=step, step=step, stop=epochs+1, dtype="int"))
    plt.ylim(bottom=0, top=1)
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
    """ Plots the confusion matrix """
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
    """ Shows both the confusion matrix and the loss and acc functions """
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


# show_npy('test/FREE_CRUISING/FREE_CRUISING_9.npy', 15)

# video_to_jpges_and_npys("data/video.avi", "data/video/")

# video_to_jpges("/Users/manuel/Dropbox/_data/01_5fps.m4v", "/Users/manuel/Dropbox/_data/01/")
