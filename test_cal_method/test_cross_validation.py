import random
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10
from KNN_Classifier import KNearestNeighbor

# Load the raw CIFAR-10 data.
cifar10_dir = 'D:/dataset/cifar-10-python/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# train numbers
num_train = 10000
X_train = X_train[:num_train]
y_train = y_train[:num_train]

# test numbers
num_test = 1000
X_test = X_test[:num_test]
y_test = y_test[:num_test]

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

KNN = KNearestNeighbor()

num_folds = 5    # split the training dataset to 5 parts
k_classes = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]    # all k, determine the best k

# Split up the training data into folds
X_train_folds = []
y_train_folds = []
X_train_folds = np.split(X_train, num_folds)
y_train_folds = np.split(y_train, num_folds)

# A dictionary holding the accuracies for different values of k
k_accuracy = {}

for k in k_classes:
    accuracies = []
    #knn = KNearestNeighbor()
    for i in range(num_folds):
        Xtr = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        ytr = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
        Xcv = X_train_folds[i]
        ycv = y_train_folds[i]
        KNN.train(Xtr, ytr)
        ycv_pred = KNN.predict(Xcv, k=k)
        accuracy = np.mean(ycv_pred == ycv)
        accuracies.append(accuracy)
    k_accuracy[k] = accuracies

# Print the accuracy
for k in k_classes:
    for i in range(num_folds):
        print('k = %d, fold = %d, accuracy: %f' % (k, i+1, k_accuracy[k][i]))


# Plot the cross validation
k_classes = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
for k in k_classes:
    plt.scatter([k] * num_folds, k_accuracy[k])
# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = [np.mean(k_accuracy[k]) for k in k_accuracy]
accuracies_std = [np.std(k_accuracy[k]) for k in k_accuracy]
plt.errorbar(k_classes, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()