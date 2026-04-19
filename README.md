# PC03 - Competition Repo

This repository will be used for tracking your model performance. Try to keep your source code here as well. The repository will be pulled to get your predictions on the specified deadlines in Moodle.

Your predictions should be placed in the `Data-Predictions` subdirectory of this repo. When you first pull the data, you will find dummy files with only label `0` in them. They have the correct format but do not have the correct predictions. You are expected to overwrite these files. The timestamps for each file is in the associated `h5` file in the `Pre\Test` dataset. The length of the predictions should match the length of the timestamps.

The file `test_format.py` contains tests for checking that your data is formatted correctly.
