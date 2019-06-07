import sys
import numpy as np
def main():
    # Open files
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    file_x = np.loadtxt("train_x")
    file_y = open(arg2)
    file_test = open(arg3)
    # Read content from file
    DataSet_X = file_x.read()
    DataSet_Y = file_y.read()
    Test_X = file_test.read()
    # Close files
    file_x.close()
    file_y.close()
    file_test.close()


if __name__ == '__main__':
    main()