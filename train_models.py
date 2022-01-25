'''
Description
---
This function executes a nested cross-validation.

Input
---
param experiment_type: type of experiment to run (idiographic or nomothetic)
param test_size: number of individuals in the test set.
param data_output_path: location of non-splitted data
param features: features to include in X_train, X_test
param targets: targets to include in y_train, y_test

Output
---
The output of this script is four files: X_train.csv, X_test.csv, y_train.csv, and y_test.csv. 

'''

from idiographic_experiment import idiographic_experiment
from nomothetic_experiment import nomothetic_experiment
import pandas as pd
                
def train_models(study_parameters):
    
    # Read files
    X   = pd.read_csv("X.csv", index_col = 0) 
    y   = pd.read_csv("y.csv", index_col = 0)
    ids = pd.read_csv("ids.csv", index_col = 0) 
    
    # Select experiment
    if study_parameters["experiment_type"] == "nomothetic":
        nomothetic_experiment(X, y, ids, study_parameters)
            
    elif study_parameters["experiment_type"] == "idiographic":
        idiographic_experiment(X, y, ids, study_parameters)
        
    else:   
        print("Experiment type", "'", experiment_type, "'", "not recognized. Choose nomothetic or idiographic.")