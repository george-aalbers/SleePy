# SleePy: Estimating sleep duration from smartphone application log data

This Github repository contains code for the project "SleePy: An open-source machine learning tool for estimating sleep duration from smartphone usage log data". 

## SleePy
This folder contains an iJupyter notebook with an example of how to run SleePy (imported from a .py file called sleepy.py) with a trained random forest (loaded from a pickle file called model.pkl).

## Code
- transform_mobiledna.py transforms the raw smartphone application usage log data into a format that allows us to downsample the smartphone log data to 15-minute time windows (i.e., total time spent on the smartphone per 15-minute time window). 
- clean_mobiledna.py cleans the resulting data. 
- create_directory.py builds a directory for storing data and pickled models. 
- do_study.py (1) preprocesses the data, (2) splits the data, (3) trains models, (4) evaluates models, and (5) creates tables and visualisations. 

## Data
Due to the sensitive nature of the raw data, we only provide preprocessed data that were used to train and test models. 

- data.csv is the full unsplitted dataset
- ids.csv contains the participant IDs associated with each row of data.csv
- X.csv contains the features associated with each row of data.csv
- y.csv contains the targets associated with each row of data.csv
- feature_names.csv contains the names of all the features 

## Experiments
The folders experiment-1 through experiment-4 contain splitted data and pickled models. Suffixes indicate which models belong to which dataset. For instance, lasso_1.pkl (found in the "models" folder of "experiment-1") is the model that is trained on X_train_1.csv and y_train_1.csv and tested in X_test_1.csv and y_test_1.csv (found in the "data" folder of "experiment-1").