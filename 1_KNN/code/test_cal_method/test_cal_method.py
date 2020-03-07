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
KNN.train(X_train, y_train)

k_arr = [1,2,5,10,15]
cal_arr = [0,1,2]

print('num_train:%d, num_test:%d' % (num_train, num_test))
for i in range(len(k_arr)):
    for j in range(len(cal_arr)):
        y_pred = KNN.predict(X_test, k=k_arr[i], cal_method=cal_arr[j])
        num_correct = np.sum(y_pred == y_test)
        accuracy = np.mean(y_pred == y_test)
        print('k=%d, cal_method=%d' %(k_arr[i], cal_arr[j]))
        print('Correct %d/%d: The accuracy is %f\n' % (num_correct, X_test.shape[0], accuracy))