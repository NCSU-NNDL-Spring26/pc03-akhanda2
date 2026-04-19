import os
import numpy as np

# List of expected files length
fileLength = {'s11_trial1.csv': 7146625, 's11_trial2.csv': 6924950, 's11_trial3.csv': 7045450,
              's12_trial1.csv': 7676148, 's12_trial2.csv': 7512373, 's12_trial3.csv': 7438071,
              's13_trial1.csv': 7531470, 's13_trial2.csv': 8516621, 's13_trial3.csv': 7428557}

fileList = list(fileLength.keys())

# Getting the path to the Trial directory
script_dir = os.path.dirname(os.path.abspath(__file__))
pathTrial = os.path.join(script_dir, 'Data-Predictions')

def test_files():
    ''' This test checks that the expected files are present in the `pathTrial` directory. It
    compares the set of expected file names with the set of actual file names in the directory. 
    '''

    # Making sure that the list of files in the `pathTrial` directory includes the fileList
    assert set(fileList) == set(os.listdir(pathTrial)), "The expected list of files does not match."

def test_fileFormat():
    ''' This test checks that each file in the `fileList` has the expected format. It checks that
    each file has only one column, the expected number of rows, and that it only contains values
    between 0 and 3 (inclusive). It also checks that each file does not only predict a single
    label.'''

    for file in fileList:
        file_path = os.path.join(pathTrial, file)
        # Loading the file as a numpy array to check its shape and data type
        data = np.loadtxt(file_path, delimiter=',', dtype=int)

        # Making sure that it has only 1 column and the expected number of rows.
        assert data.shape == (fileLength[file],), f"{file} does not have the expected shape."

        # Making sure that it only has values between 0 and 3, and that it does not only predict a single label.
        assert np.all((data >= 0) & (data <= 3)), f"{file} contains values outside the range of 0 to 3."
        assert len(np.unique(data)) > 1, f"{file} contains only a single label: {np.unique(data)[0]}."
