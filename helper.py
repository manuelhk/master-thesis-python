import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import itertools


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


def show_npy(path, number_of_images=3, title="No Title"):
    array = np.load(path)
    index = 1
    for i in range(0, 15, int(15/number_of_images)):
        plt.subplot(1, number_of_images, index)
        plt.imshow(array[i])
        plt.xlabel(path + " (" + str(i) + ")")
        index += 1
    plt.suptitle(title)
    plt.show()
    pass


def show_training_history(history_path):
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
    plt.ylim(bottom=0, top=1.7)
    plt.xlim(left=0, right=epochs-1)
    plt.legend(["Train", "Validation"])
    plt.show()
    fig.savefig("output/loss.png")
    pass


def show_confusion_matrix(y_true, y_pred, label_names, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig.savefig("output/confusion_matrix.png")
    pass


"""
show_npy('test/FREE_CRUISING/FREE_CRUISING_9.npy', 15)

video_to_jpges_and_npys("data/video.avi", "data/video/")

show_training_history("/Users/manuel/Dropbox/_data/_models/1109_v3_lstm/history.npy")

y_true = np.load("/Users/manuel/Dropbox/_data/_models/1109_v3_lstm/labels_test_data.npy")
y_pred = np.argmax(np.load("/Users/manuel/Dropbox/_data/_models/1109_v3_lstm/predictions_test_data.npy"), 1)
settings = np.load("/Users/manuel/Dropbox/_data/_models/1109_v3_lstm/settings.npy")
settings = settings.item()
label_names = settings["scenarios"]
helper.show_confusion_matrix(y_true, y_pred, label_names)
"""
