import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import io

from data_utils import load_CIFAR10_raw
from softmax_without_regression import Softmax


def pre_dataset():
    cifar10_dir = 'D:/dataset/cifar-10-python/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10_raw(cifar10_dir)
    train_num = 50000
    test_num = 10000
    X_train = X_train[0:train_num]
    y_train = y_train[0:train_num]
    X_test = X_test[0:test_num]
    y_test = y_test[0:test_num]
    # 提取hog特征
    X_train_hog = []
    for i in range(X_train.shape[0]):
        curr_features = hog(X_train[i], orientations=9, pixels_per_cell=(4, 4),
                            cells_per_block=(2, 2), visualize=False)
        X_train_hog.append(curr_features)
        if(i % 100 == 0):
            print('hog processing train_data %d' % i)
    X_train_hog = np.array(X_train_hog)

    X_test_hog = []
    for i in range(X_test.shape[0]):
        curr_features = hog(X_test[i], orientations=9, pixels_per_cell=(4, 4),
                            cells_per_block=(2, 2), visualize=False)
        X_test_hog.append(curr_features)
        if(i % 100 == 0):
            print('hog processing test data %d' % i)
    X_test_hog = np.array(X_test_hog)

    # add a parameter for W
    X_train = np.hstack([X_train_hog, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test_hog, np.ones((X_test.shape[0], 1))])

    return X_train, y_train, X_test, y_test


def get_softmax_model(X, y, learning_rate, batch_num, num_iter):
    softmax = Softmax()
    loss_history = softmax.train(X, y, learning_rate, batch_num, num_iter)
    VisualizeLoss(loss_history)
    return softmax


def VisualizeLoss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()


def pca_dim_reduction(X_train,X_test): #降维
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    X_train_transformed = pca.transform(X_train)
    X_test_transformed = pca.transform(X_test)
    return X_train_transformed,X_test_transformed

X_train, y_train, X_test, y_test = pre_dataset()

#pca降维
X_train,X_test = pca_dim_reduction(X_train,X_test)
print("reduction",X_train.shape)

batch_nums = [1000]
learning_rates = [0.001]
num_iter = 50000
for learning_rate in learning_rates:
    for batch_num in batch_nums:
        softmax = get_softmax_model(
            X_train, y_train, learning_rate, batch_num, num_iter)
        y_pred = softmax.predict(X_test)
        acc = np.mean(y_pred == y_test)
        print('learning_rate: %f;batch_num: %d; Accuracy: %f' %
              (learning_rate, batch_num, acc))