import unittest
import csv

class Test(unittest.TestCase):

    def test_read_csv_file(self):
       with open('D:PATHfile-without-header.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            print(row)

if __name__ == "__main__":
    unittest.main()