'''
Description
---
This function conducts an entire preregistered ML study from scratch. It takes as input the parameter specification for the entire study as well as path strings for 
the location of log and experience sampling data. It then conducts a study, which entails outputting an entire directory of extracted features, trained models, model 
evaluation, summaries and visualisations of results.

Input
---
param study_specification: path to a csv with n rows specifying n experiments with different parameter settings
param constants_path: path to a csv containing a constant path to the log data and ESM data

Output
---
The output of this script is an entire directory with feature extraction, model training, model evaluation, summary of results, and visualization of results.
---
'''

# Import required modules
import pandas as pd
import os
from study_parameters import study_parameters
from create_directory import create_directory
from preprocessing import preprocess, select_features_targets_ids, scale_features, order_features
from train_models import train_models
from mae import mae_nomothetic_models, mae_idiographic_models
from spearman_rho import rho_nomothetic_models, rho_idiographic_models

# Specify the study parameters
print("Specifying the study parameters")
study_parameters()
print("===")
print("===")
print("===")

# Read study specification
print("Reading study specification")
study_parameters = pd.read_json("study_parameters.json")
print("===")
print("===")
print("===")

# Preprocess data
if "data.csv" not in os.listdir():
    preprocess()
else:
    pass
    
# Split data into X, y, ids
if "X.csv" not in os.listdir():
    select_features_targets_ids(study_parameters.iloc[0,:])

    # Scale features
    scale_features()

    # Order features
    order_features()
else:
    pass
    
# Create directory for number of experiments
print("Creating folder structure")
create_directory(study_parameters.shape[0])
print("===")
print("===")
print("===")

# Train all models
print("Conducting experiments")
for experiment in study_parameters.iterrows():
    print("===")
    print("===")
    print("===")
    print("Experiment #", experiment[1]["experiment"], sep = "")
    print("===")
    print("===")
    print("===")
    train_models(experiment[1])

print("Model training has finished.")

# Evaluating results
mae_nomothetic_models()
mae_idiographic_models()
rho_nomothetic_models()
rho_idiographic_models()
    
# Print message that pipeline has finished
print("Pipeline has finished.")