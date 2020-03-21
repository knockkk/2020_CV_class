import numpy as np
import matplotlib.pyplot as plt


class BPNetWork(object):
    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.W3 = None

    def train(self, X, y, learning_rate, batch_num, num_iter, reg, dim1=15, dim2=10):
        num_train = X.shape[0]
        num_dim = X.shape[1]
        num_classes = np.max(y) + 1

        W1_dim1 = num_dim
        W1_dim2 = dim1
        W2_dim1 = dim1
        W2_dim2 = dim2
        W3_dim1 = dim2
        W3_dim2 = num_classes

        if self.W1 is None:
            self.W1 = 0.001 * np.random.randn(W1_dim1, W1_dim2)
        if self.W2 is None:
            self.W2 = 0.001 * np.random.randn(W2_dim1, W2_dim2)
        if self.W3 is None:
            self.W3 = 0.001 * np.random.randn(W3_dim1, W3_dim2)

        loss_history = []
        for i in range(num_iter):
            # Mini batch
            sample_index = np.random.choice(
                num_train, batch_num, replace=False)  # 随机取batch_num个索引
            X_batch = X[sample_index, :]
            y_batch = y[sample_index]

            loss, dW1, dW2, dW3 = self.BP_Network_cost_function(
                X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W1 -= learning_rate * dW1
            self.W2 -= learning_rate * dW2
            self.W3 -= learning_rate * dW3

            if(i % 1500 == 0 and learning_rate > 5e-7):
                learning_rate = learning_rate / 2

            if i % 500 == 0:
                print('Iteration %d / %d, learning rate %f , loss %f' %
                      (i, num_iter, learning_rate, loss))

        # visualize loss
        self.visualizeLoss(loss_history)

    def visualizeLoss(self, loss_history):
        plt.plot(loss_history)
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.show()

    def predict(self, X):
        z1 = X.dot(self.W1)
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2)
        a2 = np.tanh(z2)
        z3 = a2.dot(self.W3)
        exp_scores = np.exp(z3)
        probs = (exp_scores / np.matrix(np.sum(exp_scores, axis=1)).T).getA()
        return np.argmax(probs, axis=1)

    def BP_Network_cost_function(self, X, y, reg):
        # Forward propagation
        z1 = X.dot(self.W1)
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2)
        a2 = np.tanh(z2)
        z3 = a2.dot(self.W3)

        exp_scores = np.exp(z3)
        probs = (exp_scores / np.matrix(np.sum(exp_scores, axis=1)).T).getA()

        # calculate loss
        num_train = X.shape[0]

        ground_true = np.zeros(z3.shape)
        ground_true[range(num_train), y] = 1
        loss = -1 * np.sum(ground_true * np.log(probs)) + 0.5 * reg * (np.sum(self.W1 * self.W1) +
                                                                       np.sum(self.W2 * self.W2) + np.sum(self.W3 * self.W3))

        # Backpropagation
        delta3 = probs
        num_examples = X.shape[0]
        delta3[range(num_examples), y] -= 1
        dW3 = np.dot(a2.T, delta3)

        delta2 = delta3.dot(self.W3.T) * (1-np.power(a2, 2))
        dW2 = np.dot(a1.T, delta2)

        delta1 = delta2.dot(self.W2.T) * (1-np.power(a1, 2))
        dW1 = np.dot(X.T, delta1)

        # Add regularization terms
        dW3 += reg * self.W3
        dW2 += reg * self.W2
        dW1 += reg * self.W1

        return loss, dW1, dW2, dW3
