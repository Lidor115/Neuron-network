import sys
import numpy as np
import random

sizeOfTrain = 0.8


def shuffleAllData(x, y):
    zipped = list(zip(x, y))
    random.shuffle(zipped)
    new_set_x, new_set_y = zip(*zipped)
    return new_set_x, new_set_y


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
    trainSize=int(trainSize)
    dev_x, dev_y = DataSet_X[:trainSize], DataSet_Y[:trainSize]
    test_x, text_y = DataSet_X[trainSize:], DataSet_Y[trainSize:]


if __name__ == '__main__':
    main()
