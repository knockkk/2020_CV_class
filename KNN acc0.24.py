import pickle as p
import matplotlib.pyplot as plt
import numpy as np


# NearestNeighbor class
class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        Y = np.array(Y)  # 字典里载入的Y是list类型，把它变成array类型
        return X, Y


def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        label_names = p.load(f, encoding='latin1')
        names = label_names['label_names']
        return names


# load data
label_names = load_CIFAR_Labels("D:/dataset/cifar-10-python/cifar-10-batches-py/batches.meta")
imgX1, imgY1 = load_CIFAR_batch("D:/dataset/cifar-10-python/cifar-10-batches-py/data_batch_1")
imgX2, imgY2 = load_CIFAR_batch("D:/dataset/cifar-10-python/cifar-10-batches-py/data_batch_2")
imgX3, imgY3 = load_CIFAR_batch("D:/dataset/cifar-10-python/cifar-10-batches-py/data_batch_3")
imgX4, imgY4 = load_CIFAR_batch("D:/dataset/cifar-10-python/cifar-10-batches-py/data_batch_4")
imgX5, imgY5 = load_CIFAR_batch("D:/dataset/cifar-10-python/cifar-10-batches-py/data_batch_5")
Xte_rows, Yte = load_CIFAR_batch("D:/dataset/cifar-10-python/cifar-10-batches-py/test_batch")

Xtr_rows = np.concatenate((imgX1, imgX2, imgX3, imgX4, imgX5))
Ytr_rows = np.concatenate((imgY1, imgY2, imgY3, imgY4, imgY5))

nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
nn.train(Xtr_rows[:1000, :], Ytr_rows[:1000])  # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows[:100, :])  # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % (np.mean(Yte_predict == Yte[:100])))

# show a picture
image = imgX1[6, 0:1024].reshape(32, 32)
print(image.shape)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')  # 去除图片边上的坐标轴
plt.show()

image = imgX2[6, 0:1024].reshape(32, 32)
print(image.shape)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')  # 去除图片边上的坐标轴
plt.show()
image = imgX3[6, 0:1024].reshape(32, 32)
print(image.shape)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')  # 去除图片边上的坐标轴
plt.show()
image = imgX4[6, 0:1024].reshape(32, 32)
print(image.shape)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')  # 去除图片边上的坐标轴
plt.show()
image = imgX5[6, 0:1024].reshape(32, 32)
print(image.shape)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')  # 去除图片边上的坐标轴
plt.show()

