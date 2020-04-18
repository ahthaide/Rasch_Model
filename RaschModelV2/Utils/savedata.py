import numpy as np
def create_csv(data):
    import os.path

    name_of_file = input("What is the name of the file: ")
    D= "Results"

    completeName = os.path.join( D,name_of_file + ".csv")
    np.savetxt(completeName, data, delimiter=",")


def create_csv_diff(data, filename):
    np.savetxt("filename.csv", data, delimiter=",")

