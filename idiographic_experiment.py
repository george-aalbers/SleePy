import time
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import TimeSeriesSplit
from preprocessing import center
from build_model import build_model
from train_model import train_model
from write_to_file import write_to_file

def idiographic_experiment(X, y, ids, study_parameters):
    
    # Get total sample size
    sample_size = ids.nunique()

    # Set index to ids
    X.index = y.index
    
    # Ravel ids
    ids = np.ravel(ids)
    
    # Leave one participant out
    logo = LeaveOneGroupOut()

    # Select data for one participant
    for train_index, test_index in logo.split(X, y, ids):
        X_single_subject = X.iloc[test_index, :]
        y_single_subject = y.iloc[test_index]
        
        # Get participant ID
        loop_id = X_single_subject.index.unique()[0]
           
        if study_parameters["time_series_k_splits"] == 1:

            # Determine size of test set
            test_size = np.round(X_single_subject.shape[0] * study_parameters["time_series_test_size"], decimals = 0).astype(int)

            # Split the participant's data into train and test
            X_train, X_test = X_single_subject.iloc[:-test_size, :], X_single_subject.iloc[-test_size:, :]
            y_train, y_test = y_single_subject.iloc[:-test_size], y_single_subject.iloc[-test_size:]

            print("Writing data to file")
            print("===")
            print("===")
            print("===") 

            # Write to file
            write_to_file(X_train, X_test, y_train, y_test, loop_id, study_parameters)
            
            print("Building the model")
            print("===")
            print("===")
            print("===")      

            # Build model
            model = build_model(study_parameters)

            print("Training the model")
            print("===")
            print("===")
            print("===")                      

            # Train model
            train_model(model, X_train, y_train, study_parameters, loop_id)
            
        elif study_parameters["time_series_k_splits"] > 1:

            # Determine size of test set
            test_size = np.round(X_single_subject.shape[0] * study_parameters["time_series_test_size"], decimals = 0).astype(int)

            # Time series split
            tscv = TimeSeriesSplit(n_splits=study_parameters["time_series_k_splits"], test_size = test_size)

            # Split_number
            split_number = 0

            # Split that participant's data 
            for train_index_time_series_split, test_index_time_series_split in tscv.split(X_single_subject):

                split_number += 1
                loop_id = participant_id + "_" + str(split_number)

                X_train, X_test = X_single_subject.iloc[train_index_time_series_split, :], X_single_subject.iloc[test_index_time_series_split, :]
                y_train, y_test = y_single_subject.iloc[train_index_time_series_split], y_single_subject.iloc[test_index_time_series_split]

                print("Centering the data")
                print("===")
                print("===")
                print("===")          

                # Center their data
                y_train, y_test = center(y_train, y_test, study_parameters)

                # Write to file
                write_to_file(X_train, X_test, y_train, y_test, loop_id, study_parameters)
                
                print("Building the model")
                print("===")
                print("===")
                print("===")      

                # Build model
                model = build_model(study_parameters)

                print("Training the model")
                print("===")
                print("===")
                print("===")                      

                # Train model
                train_model(model, X_train, y_train, study_parameters, loop_id) 