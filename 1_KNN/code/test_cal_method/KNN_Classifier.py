import numpy as np
from scipy.spatial import distance


class KNearestNeighbor(object):
    """a KNN classifier with L2 distance"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. This is just memorizing all the training data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (num_train,) containing the training labels, 
          where y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, cal_method=0):
        """
        Test the classifier. 
        Inputs:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - cal_method: method to calculate the distance between test X and train X
        Return:
        - pred_y: Predict output y
        """
        # calculate the L2 distance between test X and train X
        if cal_method == 0:
            # no for-loop, vectorized
            dists = self.cal_dists_Eu(X)
        elif cal_method == 1:
            # one for-loop, half-vectorized
            dists = self.cal_dists_Man(X)
        elif cal_method == 2:
            # one for-loop, half-vectorized
            dists = self.cal_dists_Cosine(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % cal_method)

        # predict the labels
        num_test = X.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # the closest k distance loc
            dists_k_min = np.argsort(dists[i])[:k]
            # the closest k distance ,all labels
            close_y = self.y_train[dists_k_min]
            # [0,3,1,3,3,1] -> 3　as y_pred[i]
            y_pred[i] = np.argmax(np.bincount(close_y))

        return y_pred

    def cal_dists_Eu(self, X):
        """
        Calculate the distance with Euclidean Distance（欧式距离）
        Input:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        Return:
        - dists: The distance between test X and train X
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
        # shape (num_test, num_train)
        d1 = np.multiply(np.dot(X, self.X_train.T), -2)
        # shape (num_test, 1)
        d2 = np.sum(np.square(X), axis=1, keepdims=True)
        d3 = np.sum(np.square(self.X_train), axis=1)    # shape (1, num_train)
        dists = np.sqrt(d1 + d2 + d3)

        return dists

    def cal_dists_Man(self, X):
        """
        Calculate the distance with Manhattan Distance (曼哈顿距离)
        Input:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        Return:
        - dists: The distance between test X and train X
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
          dists[i] = np.sum(np.abs(self.X_train - X[i]), axis=1)
          if(i % 100 == 0):
            print("Manhattan progress:" ,i*100/num_test,"%")

        return dists

    def cal_dists_Cosine(self, X):
        """
        Calculate the distance with Cosine Distance（余弦距离）
        Input:
        - X: A numpy array of shape (num_test, D) containing the test data
          consisting of num_test samples each of dimension D.
        Return:
        - dists: The distance between test X and train X
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
              dists[i][j] = distance.cosine(X[i], self.X_train[j])
            if(i % 100 == 0):
              print("Cosine progress:" ,i*100/num_test,"%")

        return dists
