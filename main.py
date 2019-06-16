"""
Lidor Alis 201025160
Eldar Shalev 312349103
"""

import sys
import numpy as np
import random
from scipy.special import softmax

# Global params
sizeOfTrain = 0.8
InitH = 100  # between 10-784
LRT = 0.1  # learning rate between 0.0001, 0.001, 0.01 ...
input_size = 28 * 28
output_size = 10
epoch = 24  # we don't want over fitting


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
    matrix = np.random.randn(H_size, I_size)
    return matrix


def softMax(x):
    """Compute softmax values for each sets of scores in x.
    :param x: vector of features
    """
    return softmax(x,axis=0)


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
    temp = np.copy(y_hat)
    temp[temp==0] =1
    return -np.dot(specific_y, np.log2(temp))


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
    h = h / (np.max(h) or 1)
    z2 = np.dot(w2, h) + b2
    y_hat = softMax(z2)
    return y_hat, h, z1


def backprop(params, x, y, y_hat, loss, h, z1):
    """

    :param params: the params
    :param x: the line x
    :param y: the y according to the Y
    :param y_hat: the vector we guess of Y
    :param loss: the loss function
    :param h: the hyper parameter
    :param z1: the function we need from fast prop
    :return: w1,w1,b1,b2 after gradient
    """
    w1, b1, w2, b2 = params
    specific_y = np.zeros_like(y_hat)
    specific_y[int(y)] = 1

    # derivative loss by y_hat multiply derivative y_hat by z2 relevant for both d_W1 AND d_W2
    y_hat_new = y_hat - specific_y
    # Calc dloss_w2:
    dz2_w2 = h
    dlossW2 = np.dot(y_hat_new,dz2_w2.T)


    # Calc dloss w1:
    dz2_h1 = w2
    g = np.vectorize(DevRelu)
    dh1_z1 = g(z1)

    db2 = np.copy(y_hat_new)
    db1 = (np.dot(y_hat_new.T, w2) * dh1_z1.T).T
    dlossW1 = np.dot(db1,np.reshape(x,(-1,1)).T)

    return dlossW1, dlossW2, db1, db2


def train(params, epochs, learningRate, train_x, train_y):
    """

    :param params: the paramas w1,w2,b1,b2
    :param epochs: times to run the loop
    :param learningRate: the delta/cosnt to learn the data
    :param train_x: our train x
    :param train_y: the Y
    :return: the test trained
    """
    w1, b1, w2, b2 = params
    for i in range(epochs):
        #print ("Epoch num" + str(i))
        sum_loss = 0.0
        shuffleAllData(train_x, train_y)
        for x, y in zip(train_x, train_y):
            y_hat, h, z1 = calcProbability(params, Relu, x)
            loss = Loss(y_hat, y)
            sum_loss += loss
            dlossW1, dlossW2, db1, db2 = backprop(params, x, y, y_hat, loss, h, z1)
            w1 -= LRT * dlossW1
            w2 -= LRT * dlossW2
            b1 -= LRT * db1
            b2 -= LRT * db2
            params = w1, b1, w2, b2


def Dsoftmax(x):
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def get_accuracy(w1, w2, b1, b2, x_valid, y_valid):
    true = 0
    params = [w1, b1, w2, b2]
    for x, y in zip(x_valid, y_valid):
        x = np.reshape(x, (1, input_size))
        y_hat, h, z1 = calcProbability(params, Relu, x)
        max_y = y_hat.argmax(axis=0)
        if max_y[0] == int(y):
            true += 1
    return true / float(len(y_valid))

def testingDev(Dev_X,params):
    file = open("test_y.txt", 'w+')
    for i in Dev_X:
        y_hat, h, z1 = calcProbability(params, Relu, i)
        file.write(str(np.argmax(y_hat))+ "\n")
    file.close()

def main():
    # Open files and read content
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    # Parsing data to ndArrays
    DataSet_X = np.loadtxt(arg1)
    DataSet_Y = np.loadtxt(arg2)
    Dev_X = np.loadtxt(arg3)
    DataSet_X = np.divide(DataSet_X, 255)
    # Shuffle and split data to train and test
    DataSet_X, DataSet_Y = shuffleAllData(DataSet_X, DataSet_Y)



    # Declarations of Hyper-Params
    hidden_size = InitH

    W1, b1 = initialize_weights(hidden_size, input_size), initialize_weights(hidden_size)
    W2, b2 = initialize_weights(output_size, hidden_size), initialize_weights(output_size)
    params = [W1, b1, W2, b2]

    # The train algorithm
    train(params, epoch, LRT, DataSet_X, DataSet_Y)
    # For accuracy purpose
    #accuracy = get_accuracy(W1, W2, b1, b2, test_x, test_y)
    #print(epoch, accuracy * 100)
    testingDev(Dev_X, params)


if __name__ == '__main__':
    main()
