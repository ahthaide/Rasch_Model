import os
import numpy as np


def openfiletest():
    user_input = input("Enter the path of your file: ")

    assert os.path.exists(user_input), "I did not find the file at, " + str(user_input)
    f = open(user_input, 'r+')

    print("File found!")
    file = np.loadtxt(f, dtype=np.int32)
    return file
