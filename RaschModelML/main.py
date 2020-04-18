from loadmodels import loaddiffi, loadlogit, arraydif, arraylogit, loadtheta, arraytheta
from data.dataManipulation import transpose
from data.ReadFile import openfiletest, openfile
from data.CreateCsvFile import create_csv
class main:
    data = openfiletest()

    # traspose data for  diff
    diff_data = transpose(data)


    # load diff model
    loaddiffi(diff_data)

    # load data to save to file
    difficulty = arraydif(diff_data)

    # load logits
    loadlogit(data)

    # load data to save to file

    logs = arraylogit(data)

    # create scv file for difficulty
    print("Creating csv file for difficulties")
    create_csv(difficulty)

    # create csv file for logits
    print("Creating csv file for logits")
    create_csv(logs)

    # predict theta
    print("Path to file")
    data_theta = openfile()

    # load data
    loadtheta(data_theta)

    # load data to save to file

    theta = arraytheta(data_theta)

    # create scv file for theta
    print("Creating csv file for theta")
    create_csv(theta)
