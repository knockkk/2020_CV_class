import numpy as np


class Softmax(object):
    def __init__(self):
        # C * D
        self.W = None

    def train(self, X, y, learning_rate, batch_num, num_iter):
        num_train = X.shape[0]
        num_dim = X.shape[1]
        num_classes = np.max(y) + 1

        if self.W is None:
            self.W = 0.001 * np.random.randn(num_classes, num_dim)

        loss_history = []

        # learning_rate = 0.01
        for i in range(num_iter):
            # Mini batch
            sample_index = np.random.choice(num_train, batch_num, replace=False) #随机取batch_num个索引
            X_batch = X[sample_index, :]
            y_batch = y[sample_index]

            loss, gradient = self.softmax_cost_function(X_batch, y_batch)
            loss_history.append(loss)

            self.W -= learning_rate * gradient

            # if(i%2000 == 0 and learning_rate>1e-4):
            #     learning_rate =  learning_rate / 2
            if i % 500 == 0:
                print('Iteration %d / %d, learning rate %f : loss %f' % (i, num_iter,learning_rate, loss))

        return loss_history

    def predict(self, X):
        scores = X.dot(self.W.T)

        y_pred = np.zeros(X.shape[0])
        y_pred = np.argmax(scores, axis=1)

        return y_pred

    def softmax_cost_function(self, X, y):
        # X: N * D, y: N
        num_train = X.shape[0]
        scores = X.dot(self.W.T)
        exp_scores = np.exp(scores)  # N * C
        pro_scores = (exp_scores / np.matrix(np.sum(exp_scores, axis=1)).T).getA()

        ground_true = np.zeros(scores.shape)
        ground_true[range(num_train), y] = 1

        reg = 0.1
        loss = -1 * np.sum(ground_true * np.log(pro_scores))  / num_train 
        gred = -1 * (ground_true - pro_scores).T.dot(X)  / num_train 
        # loss = -1 * np.sum(ground_true * np.log(pro_scores))  / num_train   + 0.5 *reg* np.sum(self.W * self.W)
        # gred = -1 * (ground_true - pro_scores).T.dot(X)  / num_train + reg*self.W
        return loss, gred