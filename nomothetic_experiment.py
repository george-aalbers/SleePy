from sklearn.model_selection import GroupKFold
from preprocessing import center, flatten_targets
from write_to_file import write_to_file
from build_model import build_model
from train_model import train_model

def nomothetic_experiment(X, y, ids, study_parameters):
    
    # For each iteration in the outer loop, we leave out nomothetic_test_size individuals.
    group_kfold = GroupKFold(n_splits = study_parameters["outer_loop_cv_k_folds"])
    
    # Set loop id to zero
    loop_id = 0
    
    # We split the data, leaving out nomothetic_test_size individuals each time.
    for train_index, test_index in group_kfold.split(X, y, ids):
        
        # Split into train and test data
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Update the loop id
        loop_id += 1
        
        print("Training the model")
        print("===")
        print("===")
        print("===")      
        
        # Flatten the targets
        y_train = flatten_targets(y_train)
        y_test = flatten_targets(y_test)

        print("Writing data to file")
        print("===")
        print("===")
        print("===") 
        
        # Write all data to file
        write_to_file(X_train, X_test, y_train, y_test, loop_id, study_parameters)
        
        # Build model
        model = build_model(study_parameters)
        
        # Train model
        train_model(model, X_train, y_train, study_parameters, loop_id)