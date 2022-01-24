##################
# Import modules #
##################

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

##############################
# Define train/test function #
##############################

def idiographic_model(data, targets, features):
    
    # Loop through feature sets
    for feature_set_number, feature_set in enumerate([features[:60], features[60:], features]):
    
        # Loop through targets
        for target in targets[6:]:

            # For each new target, open new dataframe
            performance_single_target = pd.DataFrame()
            predictions_multiple_subjects = pd.DataFrame()

            # Then loop through participants
            for pp in data.id.unique().tolist():

                print("Currently training and testing", target, "for", pp)

                # Select data for training and testing
                data_single_subject = data[data.id.isin([pp])]
                n = data_single_subject.shape[0]-data_single_subject.shape[0]//5
                train_set = data_single_subject.iloc[:n,1:]
                test_set = data_single_subject.iloc[n:,1:]

                # Train-test split
                X_train = train_set[feature_set]
                y_train = train_set[target]
                X_test = test_set[feature_set]
                y_test = test_set[target]

                if X_train.shape[0] > 9:

                    # Initialize RandomForestRegressor
                    forest = GradientBoostingRegressor()

                    # Run RandomizedSearchCV
                    model = RandomizedSearchCV(forest, random_grid, cv=5, n_iter=100, n_jobs=64, random_state = 0)

                    # Fit the model
                    model.fit(X_train, y_train)

                    # Save model
                    with open('/home/haalbers/sleep-estimation/gen/analysis/output/idiographic/models/xgboost_' + target + "_" + pp +  "_" + str(feature_set_number) + '.pkl','wb') as f:
                        pickle.dump(model,f)

                    # Calculate performance of baseline model and idiographic model
                    baseline_train = mean_absolute_error(y_train, np.repeat(y_train.mean(),y_train.shape[0]))
                    accuracy_train = mean_absolute_error(y_train, model.predict(X_train))
                    baseline_test = mean_absolute_error(y_test, np.repeat(y_train.mean(),y_test.shape[0]))
                    accuracy_test = mean_absolute_error(y_test, model.predict(X_test))

                    # Save results in one dataframe and write to file
                    performance = pd.DataFrame({"baseline_train": baseline_train,
                                                "accuracy_train": accuracy_train,
                                                "baseline_test": baseline_test,
                                                "accuracy_test": accuracy_test,
                                                "improvement_test":((baseline_test-accuracy_test)/baseline_test)}, 
                                               index = [pp])
                    performance_single_target = pd.concat([performance_single_target, performance], axis = 0)
                    performance_single_target.to_csv("/home/haalbers/sleep-estimation/gen/analysis/output/idiographic/accuracy/xg_boost_" + target +  "_" + str(feature_set_number) + ".csv")

                    predictions_single_subject = pd.concat([y_test.reset_index(drop=True), pd.DataFrame(model.predict(X_test[feature_set]))], axis = 1)
                    predictions_single_subject.index = np.repeat(pp, predictions_single_subject.shape[0])
                    predictions_multiple_subjects = pd.concat([predictions_multiple_subjects, predictions_single_subject], axis = 0)
                    predictions_multiple_subjects.to_csv("/home/haalbers/sleep-estimation/gen/analysis/output/idiographic/predictions/xg_boost_" + target + "_" + str(feature_set_number) + ".csv")

###################################
# Hyperparameter grid for xgboost #
###################################

# Learning rate
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
# Regularization
alpha = [0.1, 1, 10, 100, 1000]
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Create the random grid
random_grid = {'learning_rate': learning_rate,
               'alpha': alpha,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

###########################
# Variable name selection #
###########################

# Read data
data = pd.read_csv("data.csv", index_col = 0)
features = pd.read_csv("features.csv", index_col = 0)
targets = pd.read_csv("targets.csv", index_col = 0)

########################
# Train and test model #
########################

idiographic_model(data, targets.iloc[:,0], features.iloc[:,0])