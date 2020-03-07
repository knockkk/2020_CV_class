import numpy as np
from data_utils import load_CIFAR10
from softmax_without_regression import Softmax
import matplotlib.pyplot as plt


def min_max_scaler(X): #归一化
    min1 = 0
    max1 = 255
    return (X-min1)/(max1-min1)

def pre_dataset():
    cifar10_dir = 'D:/dataset/cifar-10-python/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    num_train = 48000
    num_val = 2000

    mask = range(num_train, num_train + num_val)
    X_val = X_train[mask]
    y_val = y_train[mask]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    X_train = min_max_scaler(X_train)
    X_val = min_max_scaler(X_val)
    X_test = min_max_scaler(X_test)

    # add a parameter for W
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

    return X_train, y_train, X_test, y_test, X_val, y_val

def get_softmax_model(X, y,learning_rate,batch_num,num_iter):
    softmax = Softmax()
    loss_history = softmax.train(X, y, learning_rate,batch_num,num_iter)
    VisualizeLoss(loss_history)
    return softmax

def VisualizeLoss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

X_train, y_train, X_test, y_test, X_val, y_val = pre_dataset()
batch_nums = [200,300,500,1000,1500]
learning_rates = [1e-2,1e-3,1e-4]
num_iter = 2000
for learning_rate in learning_rates:
    for batch_num in batch_nums:
        softmax = get_softmax_model(X_train, y_train,learning_rate,batch_num,num_iter)
        y_pred = softmax.predict(X_test)
        acc = np.mean(y_pred == y_test)
        print('learning_rate: %f;batch_num: %d; Accuracy: %f' % (learning_rate,batch_num,acc))