import sys
import numpy as np
import scipy as sp
import random

# Global params
sizeOfTrain = 0.8
InitH = 12  # between 10-784
LRT = 0.1  # learning rate between 0.0001, 0.001, 0.01 ...
input_size = 28 * 28
output_size = 10
epoch = 10  # we don't want over fitting


def shuffleAllData(x, y):
    """
    :param x: a given data set x
    :param y: a given data set y
    :return: the new set shuffled accordingly
    """
    zipped = list(zip(x, y))
    random.shuffle(zipped)
    new_set_x, new_set_y = zip(*zipped)
    return new_set_x, new_set_y


def initialize_weights(H_size, I_size=1):
    """
    :param H_size: the hidden size - rows
    :param I_size: the columns size- default is 1
    :return: new matrix of the size we want
    """
    matrix = np.zeros(shape=(H_size, I_size))
    return matrix


def softMax(x):
    """Compute softmax values for each sets of scores in x.
    :param x: vector of features
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def Relu(x):
    """
    :param x: a given input
    :return: the number if is non-negative, else  0
    """
    return max(0, x)


def DevRelu(x):
    """
    :param x: the given line x
    :return: the derivative of the opposite relu
    """
    if x > 0: return 1
    return 0


def Loss(y_hat, y):
    """
    :param y_hat: the vector of y
    :param y: the specific y
    :return: the loss calculate
    """
    specific_y = np.zeros(y_hat.size)
    specific_y[int(y)] = 1
    # TODO CHECK ABOUT ZERO np.log2(y_hat)
    return -np.dot(specific_y, np.log2(y_hat))


def calcProbability(params, Relu, x):
    """

    :param params: the parameters w1,b1,w2,b2
    :param Relu: the function we will use
    :param x: a given row
    :return: the vector y hat probabilities and vector h
    """
    w1, b1, w2, b2 = params
    new_x = np.reshape(x, (-1, 1))
    z1 = np.dot(w1, new_x) + b1
    g = np.vectorize(Relu)
    h = g(z1)
    z2 = np.dot(w2, h) + b2
    y_hat = softMax(z2)
    return y_hat, h


def backprop(params, x, y, y_hat, loss, h):
    gradientW2 = np.outer(y_hat, h)


def train(params, epochs, learningRate, train_x, train_y, dev_x, dev_y):
    for i in range(epochs):
        sum_loss = 0.0
        shuffleAllData(train_x, train_y)
        for x, y in zip(train_x, train_y):
            y_hat, h = calcProbability(params, Relu, x)
            loss = Loss(y_hat, y)
            sum_loss += loss
            gradients = backprop(params, x, y, y_hat, loss, h)

            # w1 = w1 - LRT * w1_new


def main():
    # Open files and read content
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    # Parsing data to ndArrays
    DataSet_X = np.loadtxt(arg1)
    DataSet_Y = np.loadtxt(arg2)
    Dev_X = np.loadtxt(arg3)

    # Shuffle and split data to train and test
    DataSet_X, DataSet_Y = shuffleAllData(DataSet_X, DataSet_Y)
    trainSize = len(DataSet_X) * sizeOfTrain
    trainSize = int(trainSize)
    train_x, train_y = DataSet_X[:trainSize], DataSet_Y[:trainSize]
    test_x, test_y = DataSet_X[trainSize:], DataSet_Y[trainSize:]

    # Declarations of Hyper-Params
    hidden_size = InitH

    W1, b1 = initialize_weights(hidden_size, input_size), initialize_weights(hidden_size)
    W2, b2 = initialize_weights(output_size, hidden_size), initialize_weights(output_size)
    params = [W1, b1, W2, b2]

    # The train algorithm
    train(params, epoch, LRT, train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
