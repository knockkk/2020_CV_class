import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.misc import imread


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    print(filename)
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def load_tiny_imagenet(path, dtype=np.float32):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.
    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    Returns: A tuple of
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.iteritems():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test)

    return class_names, X_train, y_train, X_val, y_val, X_test, y_test


def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt) will
    be skipped.
    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.
    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = pickle.load(f)['model']
            except pickle.UnpicklingError:
                continue
    return models


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, num_classes, std=1e-4):
        """
        Weights are initialized to small random values and biases are initialized to zero.
        """
        self.parameters = {}
        self.parameters['W1'] = std * np.random.randn(hidden_size, input_size)
        self.parameters['b1'] = np.zeros(hidden_size)
        self.parameters['W2'] = std * np.random.randn(num_classes, hidden_size)
        self.parameters['b2'] = np.zeros(num_classes)

    def loss(self, X, y, reg):
        """
        X: N * D
        y: N * 1
        """
        W1 = self.parameters['W1']  # H * D
        b1 = self.parameters['b1']  # H * 1
        W2 = self.parameters['W2']  # C * H
        b2 = self.parameters['b2']  # C * 1
        num_examples = X.shape[0]

        # Compute the forward pass
        Relu = lambda x: np.maximum(0, x)
        z1 = X.dot(W1.T) + b1  # N * H
        a1 = Relu(z1)
        z2 = a1.dot(W2.T) + b2  # N * C
        scores = z2

        if y is None:
            return scores

        # Compute the loss
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        pro_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        ground_true = np.zeros(scores.shape)
        ground_true[range(num_examples), y] = 1
        loss = -np.sum(ground_true * np.log(pro_scores)) / num_examples + 0.5 * reg * (
                    np.sum(W1 * W1) + np.sum(W2 * W2))

        # Backward pass: compute gradients
        grads = {}
        # Compute the gradient of z2 (scores)
        dz2 = -(ground_true - pro_scores) / num_examples  # N * C
        # Backprop into W2, b2 and a1
        dW2 = dz2.T.dot(a1)  # C * H
        db2 = np.sum(dz2, axis=0)  # 1 * C
        da1 = dz2.dot(W2)  # N * H
        # Backprop into z1
        dz1 = da1
        dz1[a1 <= 0] = 0  # N * H
        # Backprop into W1, b1
        dW1 = dz1.T.dot(X)  # H * D
        db1 = np.sum(dz1, axis=0)  # 1 * H

        # add the regularization
        grads['W1'] = dW1 + reg * W1
        grads['b1'] = db1
        grads['W2'] = dW2 + reg * W2
        grads['b2'] = db2

        return loss, grads

    def train(self, X, y, X_val, y_val, reg, learning_rate,
              learning_rate_decay, iterations_per_lr_annealing,
              num_epoches, batch_size, verbose):
        num_examples = X.shape[0]
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        iterations_per_epoch = max(num_examples / batch_size, 1)
        num_iters = int(num_epoches * iterations_per_epoch)

        for i in range(num_iters):
            # mini batch
            sample_index = np.random.choice(num_examples, batch_size, replace=True)
            X_batch = X[sample_index, :]
            y_batch = y[sample_index]

            # cal loss
            loss, grads = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.parameters['W1'] -= learning_rate * grads['W1']
            self.parameters['b1'] -= learning_rate * grads['b1']
            self.parameters['W2'] -= learning_rate * grads['W2']
            self.parameters['b2'] -= learning_rate * grads['b2']

            if verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, num_iters, loss))

            if i % iterations_per_epoch == 0:
                train_acc_history.append(np.mean(self.predict(X_batch) == y_batch))
                val_acc_history.append(np.mean(self.predict(X_val) == y_val))

            if i % iterations_per_lr_annealing == 0:
                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def predict(self, X):
        # Compute the forward pass
        Relu = lambda x: np.maximum(0, x)
        z1 = X.dot(self.parameters['W1'].T) + self.parameters['b1']
        a1 = Relu(z1)
        z2 = a1.dot(self.parameters['W2'].T) + self.parameters['b2']
        score = z2

        y_pred = np.argmax(score, axis=1)
        return y_pred

def pre_dataset(path):
    X_train, y_train, X_test, y_test = load_CIFAR10(path)

    num_train = 49000
    num_val = 1000

    mask = range(num_train, num_train + num_val)
    X_val = X_train[mask]
    y_val = y_train[mask]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))

    """
    print('Train data shape: {}'.format(X_train.shape))
    print('Train labels shape: {}'.format(y_train.shape))
    print('Validation data shape: {}'.format(X_val.shape))
    print('Validation labels shape: {}'.format(y_val.shape))
    print('Test data shape: {}'.format(X_test.shape))
    print('Test labels shape: {}'.format(y_test.shape))
    """
    return X_train, y_train, X_test, y_test, X_val, y_val


def auto_get_parameters(X_train, y_train, X_val, y_val):
    "hidden layer size, learning rate, decay of learning rate, iterations_per_Annealing the learning rate, numer of training epochs, and regularization strength"
    best_net = None
    best_acc = -1
    best_parameters = None
    results = {}
    # params
    input_size = 32 * 32 * 3
    hidden_size = 80
    num_classes = 10
    learning_rate = [8e-4, 9e-4]
    reg = [0.01, 0.1]
    learning_rate_decay = [0.95, 0.97, 0.99]
    iterations_per_lr_annealing = [400, 500]  # annealing learning rate per 200 iters
    num_epoches = [15]  # a epoch is (number_of_train / batch_size)
    batch_size = [250]  # mini batch

    figure_index = 0
    for lr in learning_rate:    # 重要参数
        for rs in reg:
            for lrd in learning_rate_decay:
                for ipla in iterations_per_lr_annealing:
                    for ne in num_epoches:
                        for bs in batch_size:
                            figure_index += 1
                            print('current params: %s, %s, %s, %s, %s, %s' % (lr, rs, lrd, ipla, ne, bs))
                            net, val_acc = visualize_net(figure_index, input_size, hidden_size, num_classes, lr, rs,
                                                         lrd, ipla, ne, bs)
                            results[(lr, rs, lrd, ipla, ne, bs)] = val_acc

                            if best_acc < val_acc:
                                best_acc = val_acc
                                best_net = net
                                best_parameters = (lr, rs, lrd, ipla, ne, bs)

    for lr, rs, lrd, ipla, ne, bs in sorted(results):
        val_accuracy = results[(lr, rs, lrd, ipla, ne, bs)]
        print('lr %e, reg %e, lrd %e, ipla %e, ne %e, bs %e val accuracy: %f' % (
            lr, rs, lrd, ipla, ne, bs, val_accuracy))
    print('best validation accuracy achieved during cross-validation: {}, which parameters is {}'.format(best_acc,
                                                                                                         best_parameters))
    plt.show()
    return best_net


def visualize_net(figure_index, input_size, hidden_size, num_classes, learning_rate, reg,
                  learning_rate_decay, iterations_per_lr_annealing, num_epoches, batch_size):
    # Create a network
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
                      reg, learning_rate, learning_rate_decay,
                      iterations_per_lr_annealing, num_epoches,
                      batch_size, False)

    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()

    # Plot the loss function and train / validation accuracies
    params = {'lr': learning_rate, 'reg': reg, 'lrd': learning_rate_decay,
              'lrpla': iterations_per_lr_annealing, 'ne': num_epoches,
              'bs': batch_size, 'val_acc': ("%.2f" % val_acc)}
    plt.figure(figure_index)
    plt.figtext(0.04, 0.95, params, color='green')
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')

    return net, val_acc


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_val, y_val = pre_dataset('D:/dataset/cifar-10-python/cifar-10-batches-py/')
    best_net = auto_get_parameters(X_train, y_train, X_val, y_val)
    test_acc = np.mean(best_net.predict(X_test) == y_test)
    print('Test accuracy: {}'.format(test_acc))