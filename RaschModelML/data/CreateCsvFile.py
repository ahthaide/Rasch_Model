import numpy as np
import pandas as pd
import os


def create_csv(data):
    import os.path

    name_of_file = input("What is the Path of the file: ")
    D= "Results"

    completeName = os.path.join( D,name_of_file + ".csv")
    np.savetxt(completeName, data, delimiter=",")


def create_csv_diff(data, Path ):
    np.savetxt("filename.csv", data, delimiter=",")


