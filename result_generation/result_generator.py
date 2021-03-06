import itertools

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('Actual CS')
    plt.xlabel('Predicted CS')
    plt.savefig('../results/confusion_matrix.pdf')
    plt.clf()
    plt.cla()
    plt.close()


def process_results_for_classification(predicted_cs, actual_cs, output_shape):
    precisions, recall, f1_score, _ = mt.precision_recall_fscore_support(actual_cs, predicted_cs,
                                                                         labels=range(0, output_shape))

    print("Ground Truth CS: ", actual_cs)
    print("Predicted CS: ", predicted_cs)
    print("Precision : ", precisions)
    print("Recall: ", recall)
    print("F1-Score: ", f1_score)

    return precisions, recall, f1_score


def plot_train_vs_val_loss(model_training_history):
    plt.plot(model_training_history.history['loss'])
    plt.plot(model_training_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('../results/train_vs_val_loss.pdf')
    plt.clf()
    plt.cla()
    plt.close()


def plot_train_vs_val_acc(model_training_history):
    plt.plot(model_training_history.history['accuracy'])
    plt.plot(model_training_history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('../results/train_vs_val_acc.pdf')
    plt.clf()
    plt.cla()
    plt.close()
