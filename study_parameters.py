'''
Description
---
This function generates a .json file containing instructions for the machine learning pipeline. 

'''

import pandas as pd
import numpy as np
import os

def study_parameters():
    
    # Root folder for the directory structure
    root_folder = os.getcwd()     

    # Number of experiments
    n_experiments = 8
    
    # Features
    features = {'21:00:00', '21:15:00', '21:30:00', '21:45:00', '22:00:00', 
                '22:15:00', '22:30:00', '22:45:00', '23:00:00', '23:15:00', 
                '23:30:00', '23:45:00', 
                '00:15:00.1', '00:30:00.1', '00:45:00.1', '01:00:00.1',
                '01:15:00.1', '01:30:00.1', '01:45:00.1', '02:00:00.1',
                '02:15:00.1', '02:30:00.1', '02:45:00.1', '03:00:00.1',
                '03:15:00.1', '03:30:00.1', '03:45:00.1', '04:00:00.1',
                '04:15:00.1', '04:30:00.1', '04:45:00.1', '05:00:00.1',
                '05:15:00.1', '05:30:00.1', '05:45:00.1', '06:00:00.1',
                '06:15:00.1', '06:30:00.1', '06:45:00.1', '07:00:00.1',
                '07:15:00.1', '07:30:00.1', '07:45:00.1', '08:00:00.1',
                '08:15:00.1', '08:30:00.1', '08:45:00.1', '09:00:00.1',
                '09:15:00.1', '09:30:00.1', '09:45:00.1', '10:00:00.1',
                '10:15:00.1', '10:30:00.1', '10:45:00.1', '11:00:00.1',
                '11:15:00.1', '11:30:00.1', '11:45:00.1', '12:00:00.1',
                '12:15:00.1', '12:30:00.1', '12:45:00.1', '13:00:00.1'}
    
    # Targets
    targets = "self_reported_sleep_duration"
    
    # Model types in this study 
    models = ["lasso", "svr", "rf", "gbr", "lasso", "svr", "rf", "gbr"]
    
    # Hyperparameters of the models we train in the study
    lasso_parameters = {"alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
    
    svm_parameters = {"kernel": ["rbf"],
                      "gamma": ["scale","auto"],
                      "C": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000],
                      "epsilon": [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1]}
    
    rf_parameters = {'n_estimators': [500, 1000, 2000],
                     'min_samples_leaf': [0.001, 0.01], 
                     'min_samples_split': [0.001, 0.01],
                     'min_weight_fraction_leaf': [0.001, 0.01],
                     'max_features': np.arange(2, 12, 5).tolist(),
                     'max_leaf_nodes' : [2, 5, 10],
                     'max_depth' : [1, 5, 10]
                     }

    gbr_parameters = {'learning_rate': np.arange(0.1,0.9,0.1),
                   'alpha': np.arange(0.1,0.9,0.1),
                   'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4]}
    
    model_parameters = [lasso_parameters, svm_parameters, rf_parameters, gbr_parameters, lasso_parameters, svm_parameters, rf_parameters, gbr_parameters]
    
    # Dataframe containing instructions for the study
    study_parameters = pd.DataFrame({"esm_data_path":             np.repeat('/home/haalbers/dissertation/experience-sampling-clean.csv', n_experiments),
                                     "log_data_path":             np.repeat("/home/haalbers/dissertation/mobiledna-clean.csv", n_experiments),
                                     "data_output_path":          (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "data/").values,
                                     "model_output_path":         (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "models/").values,
                                     "results_output_path":       (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "results/").values,
                                     "explanations_output_path":  (root_folder + "/experiment-" + pd.Series(range(1, n_experiments + 1, 1)).astype(str) + "/" + "explanations/").values,
                                     "baseline_path":             np.repeat(root_folder + "/baseline/", n_experiments),
                                     "markdown_path":             np.repeat(root_folder + "/markdown/", n_experiments),
                                     "id_variable":               np.repeat("id", n_experiments),
                                     "features":                  np.tile(features, n_experiments),
                                     "targets":                   np.tile(targets, n_experiments),                                     
                                     "experiment":                range(1, n_experiments + 1),
                                     "experiment_type":           ["nomothetic", "nomothetic", "nomothetic", "nomothetic", "idiographic", "idiographic", "idiographic", 
                                                                   "idiographic"],
                                     "window_size":               np.repeat(60, n_experiments),
                                     "prediction_task":           np.repeat("regression", n_experiments),
                                     "cross_validation_type":     np.repeat("grid", n_experiments),
                                     "outer_loop_cv_k_folds":     np.repeat(5, n_experiments),
                                     "inner_loop_cv_k_folds":     np.repeat(5, n_experiments),
                                     "time_series_k_splits":      np.repeat(1, n_experiments),
                                     "time_series_test_size":     np.repeat(0.2, n_experiments),
                                     "n_jobs":                    np.repeat(64, n_experiments),
                                     "model_type":                models,
                                     "model_parameters":          model_parameters}, 
                                    index = np.arange(n_experiments))
    
    # Write this dataframe to .json
    study_parameters.to_json("study_parameters.json")
    
study_parameters()