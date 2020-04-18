# imports
import pandas as pd
import sys
import os


def openfiletest():
    user_input = input("Enter the path of your file: ")

    assert os.path.exists(user_input), "I did not find the file at, " + str(user_input)
    f = open(user_input, 'r+')

    print("File found!")
    file = pd.read_csv(f)
    return file
def openfile():

    filename = input ("filename: ")
    with open (filename) as f:
        file =pd.read_csv(filename)
        return file

