# Import modules
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

# Define function for creating hyperparameter grid for random forest
def hyperparameter_grid():
    
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
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    return random_grid

# Define function that returns dataframe with y_true and y_pred
def make_predictions(X_test, y_test, forest, test_ids):
    
    test_predictions = pd.DataFrame(forest.predict(X_test))    
    test_predictions = pd.concat([y_test.reset_index(drop=True), test_predictions.reset_index(drop=True)], axis = 1)
    test_predictions.columns = ["y_test","y_pred"]
    test_predictions.index = test_ids
    test_predictions.reset_index(inplace=True)
    
    return test_predictions

# Define function for evaluating the model per participant
def evaluate_predictions(test_predictions, y_train, test_ids):

    y_test = test_predictions.y_test
    y_pred = test_predictions.y_pred
    y_test.index = test_predictions.id
    y_pred.index = test_predictions.id
    
    results_multiple_subjects = pd.DataFrame()
    
    for p in test_ids.unique().tolist():
        
        y_test_pp = y_test[y_test.index == p]
        y_pred_pp = y_pred[y_pred.index == p]
        
        baseline_test = mean_absolute_error(y_test_pp, np.repeat(y_train.mean(), y_test_pp.shape[0]))
        accuracy_test = mean_absolute_error(y_test_pp, y_pred_pp)

        results_single_subject = pd.DataFrame({"baseline_test":baseline_test, "accuracy_test":accuracy_test}, index = [p])
        results_multiple_subjects = pd.concat([results_multiple_subjects, results_single_subject], axis = 0)
    
    results_multiple_subjects = pd.DataFrame(results_multiple_subjects)
    
    return results_multiple_subjects
    
# Define function for training the random forest
def train_nomothetic_rf(data, features, targets, userlist):
        
    for feature_set_number, feature_set in enumerate([features[:60], features[60:], features]):

        accuracies = pd.DataFrame()
        
        for n, pp in enumerate(userlist):

            # Select participants for training and testing
            pp_train = ~data.id.isin(pp)
            pp_test = data.id.isin(pp)
            train_set = data[pp_train]
            test_set = data[pp_test]
            train_ids = train_set.id
            test_ids = test_set.id

            # Hyperparameter grid
            random_grid = hyperparameter_grid()

            # Loop through targets and train model for each
            for target in targets[6:]:

                # Split into features and targets for train and test data
                X_train = train_set[feature_set]
                y_train = train_set[target]
                X_test = test_set[feature_set]
                y_test = test_set[target]

                # Initialize random forest
                forest = RandomForestRegressor(criterion='mae')

                # Do randomized search CV with group k-fold cross-validation
                gkf = GroupKFold(n_splits = 5)
                forest = RandomizedSearchCV(forest, random_grid, cv=gkf, n_iter=5, verbose=5, n_jobs=64, random_state = 0)
                ids = train_set.id

                # Fit the model
                forest.fit(X_train, y_train, groups = ids)

                # Save model
                with open('/home/haalbers/sleep-estimation/gen/analysis/output/nomothetic/models/nomothetic-rf-' + target + "_CV_" + str(n) + "_FEATURE_SET_" + str(feature_set_number) + '.pkl','wb') as f:
                    pickle.dump(forest,f)

                # Make predictions
                test_predictions = make_predictions(X_test, y_test, forest, test_ids)
                
                # Write predictions to file
                test_predictions.to_csv("/home/haalbers/sleep-estimation/gen/analysis/output/nomothetic/predictions/rf-" + target + "_CV_" + str(n) + "_FEATURE_SET_" + str(feature_set_number) + ".csv")
                
                # Evaluate performance
                test_performance = evaluate_predictions(test_predictions, y_train, test_ids)
                
                # Concatenate results to 'accuracies' object
                accuracies = pd.concat([accuracies, test_performance], axis = 0)

                # Write test results to file
                accuracies.to_csv("/home/haalbers/sleep-estimation/gen/analysis/output/nomothetic/accuracy/model-performance-rf-" + target + "_FEATURE_SET_" + str(feature_set_number) + ".csv")

# Read data
data = pd.read_csv("/home/haalbers/sleep-estimation/src/data.csv", index_col = 0)

# Get feature and target names
features = sum(pd.read_csv("features.csv", index_col = 0).values.tolist(), [])
targets = sum(pd.read_csv("targets.csv", index_col = 0).values.tolist(), [])

# User list for dropping participants from train/validation set (these participants are in the test set)
userlist =  [['User #23877',
              'User #25163',
              'User #24324',
              'User #24035',
              'User #17984',
              'User #23921',
              'User #15171',
              'User #15510',
              'User #24127',
              'User #14995',
              'User #15044',
              'User #24426',
              'User #24249',
              'User #15024',
              'User #23743',
              'User #24235',
              'User #23756',
              'User #24612',
              'User #24121',
              'User #23881',
              'User #24198',
              'User #23876',
              'User #25189',
              'User #23983',
              'User #23926',
              'User #15373',
              'User #15462',
              'User #15205',
              'User #15159',
              'User #24083',
              'User #25172',
              'User #23955',
              'User #15382',
              'User #15405',
              'User #15280',
              'User #15242',
              'User #15208',
              'User #25250',
              'User #25164'],
             ['User #25201',
              'User #24087',
              'User #15445',
              'User #23899',
              'User #23984',
              'User #23912',
              'User #24467',
              'User #25180',
              'User #24122',
              'User #25259',
              'User #25569',
              'User #24118',
              'User #25278',
              'User #23995',
              'User #18001',
              'User #23778',
              'User #24226',
              'User #24350',
              'User #24145',
              'User #15444',
              'User #23871',
              'User #23884',
              'User #23785',
              'User #24223',
              'User #15484',
              'User #15335',
              'User #24071',
              'User #23985',
              'User #15729',
              'User #23837',
              'User #23982',
              'User #24167',
              'User #24232',
              'User #23753',
              'User #23897',
              'User #25557',
              'User #15258',
              'User #23867',
              'User #25275'],
             ['User #23767',
              'User #24007',
              'User #15020',
              'User #23770',
              'User #15277',
              'User #25208',
              'User #24041',
              'User #24033',
              'User #10599',
              'User #15403',
              'User #24124',
              'User #23768',
              'User #25522',
              'User #14981',
              'User #24201',
              'User #23745',
              'User #15588',
              'User #23988',
              'User #23906',
              'User #23913',
              'User #15031',
              'User #24207',
              'User #15173',
              'User #24346',
              'User #24206',
              'User #24213',
              'User #25235',
              'User #15115',
              'User #23829',
              'User #15448',
              'User #24197',
              'User #24196',
              'User #24248',
              'User #25169',
              'User #23883',
              'User #23961',
              'User #23878',
              'User #24470',
              'User #24224'],
             ['User #15052',
              'User #25176',
              'User #24243',
              'User #23898',
              'User #23833',
              'User #15482',
              'User #15110',
              'User #15209',
              'User #24343',
              'User #24125',
              'User #25207',
              'User #15241',
              'User #15261',
              'User #15441',
              'User #24246',
              'User #25174',
              'User #24212',
              'User #23800',
              'User #15170',
              'User #25260',
              'User #24185',
              'User #24203',
              'User #25175',
              'User #23822',
              'User #23738',
              'User #15270',
              'User #23979',
              'User #24461',
              'User #15151',
              'User #23919',
              'User #23893',
              'User #23889',
              'User #24225',
              'User #24194',
              'User #23935',
              'User #23872',
              'User #23740',
              'User #23915',
              'User #23880'],
             ['User #23875',
              'User #25197',
              'User #25178',
              'User #25547',
              'User #23953',
              'User #25498',
              'User #24464',
              'User #24227',
              'User #15530',
              'User #23914',
              'User #15203',
              'User #24199',
              'User #24348',
              'User #15304',
              'User #23870',
              'User #15368',
              'User #24079',
              'User #24349',
              'User #23924',
              'User #15051',
              'User #23909',
              'User #23737',
              'User #25171',
              'User #24193',
              'User #23746',
              'User #25203',
              'User #24120',
              'User #23787',
              'User #25167',
              'User #24081',
              'User #15503',
              'User #23981',
              'User #24347',
              'User #24040',
              'User #23987',
              'User #15419',
              'User #23769',
              'User #24192',
              'User #23917',
              'User #25211',
              'User #24195']]

# Train model
train_nomothetic_rf(data, features, targets, userlist)