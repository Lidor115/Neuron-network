import sys
import numpy as np
import random

# Global params
sizeOfTrain = 0.8
InitH = 12  # between 10-784
LRT = 0.1  # learning rate between 0.0001, 0.001, 0.01 ...
input_size = 28 * 28
output_size = 10


def shuffleAllData(x, y):
    zipped = list(zip(x, y))
    random.shuffle(zipped)
    new_set_x, new_set_y = zip(*zipped)
    return new_set_x, new_set_y


def initialize_weights(H_size, DataSet, I_size=1):
    matrix = np.zeros(shape=(H_size,I_size))
    return matrix


def main():
    # Open files and read content
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    # Parsing data to ndArrays
    DataSet_X = np.fromfile(arg1)
    DataSet_Y = np.loadtxt(arg2)
    Test_X = np.loadtxt(arg3)

    # Shuffle and split data to train and test
    DataSet_X, DataSet_Y = shuffleAllData(DataSet_X, DataSet_Y)
    trainSize = len(DataSet_X) * sizeOfTrain
    trainSize = int(trainSize)
    dev_x, dev_y = DataSet_X[:trainSize], DataSet_Y[:trainSize]
    test_x, text_y = DataSet_X[trainSize:], DataSet_Y[trainSize:]

    # Declerations of Hyper-Params
    hidden_size = InitH

    W1, b1 = initialize_weights(hidden_size, DataSet_X, input_size), initialize_weights(hidden_size, DataSet_X)
    W2, b2 = initialize_weights(output_size, DataSet_Y, hidden_size), initialize_weights(output_size, DataSet_Y)
   # w1 = w1 - LRT * w1_new


if __name__ == '__main__':
    main()
