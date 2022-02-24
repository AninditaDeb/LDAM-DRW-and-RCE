from __future__ import absolute_import
from __future__ import print_function
import os

import argparse
import matplotlib.pyplot as plt

import numpy as np

from numpy import interp
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from imblearn.over_sampling import RandomOverSampler
from sklearn import svm, metrics
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, roc_curve, auc, \
    roc_auc_score, PrecisionRecallDisplay, recall_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

from itertools import cycle

# Set random seed
np.random.seed(123)


def other_class(n_classes, current_class):
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


def get_lr_scheduler(dataset):
    if dataset in ['mnist', 'MNIST']:
        def scheduler(epoch):
            if epoch > 30:
                return 0.001
            elif epoch > 10:
                return 0.01
            else:
                return 0.1

        return LearningRateScheduler(scheduler)


NUM_CLASSES = {'mnist': 10}


def get_data(dataset, noise_ratio, asym, balanced, random_shuffle):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if balanced:
        X_train_reshaped = X_train.reshape(-1, 784)
        ros = RandomOverSampler()
        X_balanced, Y_balanced = ros.fit_resample(X_train_reshaped, y_train)
        X_train = X_balanced.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = Y_balanced
        y_train_clean = np.copy(Y_balanced)
        Y_clean_train = np.copy(Y_balanced)
        Y_clean_test = np.copy(y_test)
    else:
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        y_train = y_train
        y_train_clean = np.copy(y_train)
        Y_clean_train = np.copy(y_train)
        Y_clean_test = np.copy(y_test)

    if noise_ratio > 0:
        if asym and balanced:
            data_file = "asym_bal_%s_train_labels_%s.npy" % (dataset, noise_ratio)

        elif asym:
            data_file = "asym_imbal_%s_train_labels_%s.npy" % (dataset, noise_ratio)
        elif balanced:
            data_file = "sym_bal_%s_train_labels_%s.npy" % (dataset, noise_ratio)
        else:
            data_file = "sym_imbal_%s_train_labels_%s.npy" % (dataset, noise_ratio)
        if os.path.isfile(data_file):
            y_train = np.load(data_file)

        else:
            if asym:
                # 1 < - 7, 2 -> 7, 3 -> 8, 5 <-> 6
                source_class = [7, 2, 3, 5, 6]
                target_class = [1, 7, 8, 6, 5]
                for s, t in zip(source_class, target_class):
                    cls_idx = np.where(y_train_clean == s)[0]

                    n_noisy = int(noise_ratio * cls_idx.shape[0] / 100)

                    noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                    y_train[noisy_sample_index] = t

            else:
                n_samples = y_train.shape[0]
                n_noisy = int(noise_ratio * n_samples / 100)
                class_index = [np.where(y_train_clean == i)[0] for i in range(NUM_CLASSES[dataset])]
                class_noisy = int(n_noisy / NUM_CLASSES[dataset])

                noisy_idx = []
                for d in range(NUM_CLASSES[dataset]):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)

                for i in noisy_idx:
                    y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])

            np.save(data_file, y_train)

        # print statistics
        print("Print noisy label generation statistics:")
        for i in range(NUM_CLASSES[dataset]):
            n_noisy = np.sum(y_train == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))

    if random_shuffle:
        # random shuffle
        idx_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train, y_train_clean = X_train[idx_perm], y_train[idx_perm], y_train_clean[idx_perm]

    # one-hot-encode the labels
    y_train_clean = np_utils.to_categorical(y_train_clean, NUM_CLASSES[dataset])
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test", y_test.shape)
    Y_train = np.argmax(y_train, axis=1)
    clean_selected = np.argwhere(Y_train == Y_clean_train).reshape((-1,))
    noisy_selected = np.argwhere(Y_train != Y_clean_train).reshape((-1,))
    print("#correct labels: %s, #incorrect labels: %s" % (len(clean_selected), len(noisy_selected)))
    return X_train, y_train, X_test, y_test


def mlmodel(model_name, noise_ratio=40, noise_type='sym', dataset_type='imbal'):
    dataset = 'mnist'
    if noise_type in ['sym', 'Sym', 'SYM']:
        if dataset_type in ['bal', 'Bal', 'BAL']:
            X_train, Y_train, X_test, Y_test = get_data(dataset, noise_ratio=noise_ratio, asym=False,
                                                        balanced=True, random_shuffle=False)
        elif dataset_type in ['imbal', 'Imbal', 'IMBAL']:
            XX_train, Y_train, X_test, Y_test = get_data(dataset, noise_ratio=noise_ratio, asym=False,
                                                         balanced=False, random_shuffle=False)
    elif noise_type in ['asym', 'Asym', 'ASYM']:
        if dataset_type in ['bal', 'Bal', 'BAL']:
            X_train, Y_train, X_test, Y_test = get_data(dataset, noise_ratio=noise_ratio, asym=True,
                                                        balanced=True, random_shuffle=False)
        elif dataset_type in ['imbal', 'Imbal', 'IMBAL']:
            X_train, Y_train, X_test, Y_test = get_data(dataset, noise_ratio=noise_ratio, asym=True,
                                                        balanced=False, random_shuffle=False)
    else:
        print("Unknown arguments! Retry...")
        exit(0)

    X_train_flattened = X_train.reshape(-1, 784)
    Y_train_new = np.argmax(Y_train, axis=1)
    Y_test_new = np.argmax(Y_test, axis=1)
    X_test_flattened = X_test.reshape(-1, 784)

    if model_name in ['svm', 'Svm', 'SVM']:
        # Bagging classifier to significantly speed up classifier
        n_estimators = 10
        svc2 = BaggingClassifier(svm.SVC(gamma=0.05, C=5), bootstrap=False, max_samples=(1.0 / n_estimators),
                                 n_estimators=n_estimators, n_jobs=-1)
        svc2.fit(X_train_flattened, Y_train_new)
        Y_pred_asym_imb = svc2.predict(X_test_flattened)
        Y_score = svc2.decision_function(X_test_flattened)
    elif model_name in ['lr', 'Lr', 'LR']:
        non_linear_classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
        non_linear_classifier.fit(X_train_flattened, Y_train_new)
        Y_pred_asym_imb = non_linear_classifier.predict(X_test_flattened)
        Y_pred_asym_imb_prob = non_linear_classifier.predict_proba(X_test_flattened)
        Y_score = non_linear_classifier.decision_function(X_test_flattened)
    else:
        print('Model not implemented!')
        exit(0)

    rec = recall_score(Y_test_new, Y_pred_asym_imb, average="micro")
    print("accuracy:", metrics.accuracy_score(y_true=Y_test_new, y_pred=Y_pred_asym_imb))
    print("Recall: {}".format(rec))
    cm2 = (metrics.confusion_matrix(y_true=Y_test_new, y_pred=Y_pred_asym_imb))
    print("Confusion matrix: ")
    print(cm2)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=svc2.classes_)
    disp.plot()
    plt.show()

    # For each class
    n_classes = 10
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], Y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], Y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), Y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, Y_score, average="micro")
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot()
    _ = display.ax_.set_title("Micro-averaged over all classes")

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(
        ["aqua", "darkorange", "cornflowerblue", "darkblue", "green", "black", "violet", "yellow", "darkgreen",
         "purple", "pink"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.4f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()


def main(args):
    mlmodel(model_name=args.model_name, noise_ratio=args.noise_ratio, noise_type=args.noise_type,
            dataset_type=args.dataset_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_name',
        help="Model name: 'svm', 'lr'",
        required=True, type=str
    )
    parser.add_argument(
        '-r', '--noise_ratio',
        help="The percentage of noisy labels [0, 100].",
        required=False, type=int
    )
    parser.add_argument(
        '-t', '--noise_type',
        help="Noise type: 'sym', 'asym' ",
        required=False, type=str
    )
    parser.add_argument(
        '-d', '--dataset_type',
        help="Type of dataset: 'bal', 'imbal' ",
        required=False, type=str
    )

    parser.set_defaults(noise_ratio=0)
    parser.set_defaults(noise_type='asym')
    parser.set_defaults(dataset_type='imbal')
    args = parser.parse_args()
    # args = parser.parse_args(['-m', 'svm', '-r', '40',
    #                          '-t', 'sym', '-d', 'bal'])

    main(args)
