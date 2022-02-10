import os
import pickle
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def return_rho(df):
    rho, p = spearmanr(df.y_test, df.y_pred)
    return rho

def return_mae(df):
    mae = median_absolute_error(df.y_test, df.y_pred)
    return mae

def return_overall_rho(experiment, model_name):
    
    root = os.getcwd()
    
    multiple_rho = pd.DataFrame()
    
    for i in range(1,6,1):

        filename = root + "/experiment-" + str(experiment) + "/models/" + model_name + "_" + str(i) + ".pkl"

        model = pickle.load(open(filename, 'rb'))
        X_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/X_test_" + str(i) + ".csv", index_col = 0)
        y_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/y_test_" + str(i) + ".csv", index_col = 0)

        y_pred = model.predict(X_test)
        y_pred = pd.DataFrame(y_pred)
        y_pred.columns = ["y_pred"]
        y_test.columns = ["y_test"]

        df = pd.concat([y_test.reset_index(), y_pred], axis = 1)
        
        multiple_rho = pd.concat([multiple_rho, pd.Series(return_rho(df))], axis = 0)
    
    return multiple_rho

def return_overall_mae(experiment, model_name):
    
    root = os.getcwd()
    
    multiple_mae = pd.DataFrame()
    
    for i in range(1,6,1):

        filename = root + "/experiment-" + str(experiment) + "/models/" + model_name + "_" + str(i) + ".pkl"

        model = pickle.load(open(filename, 'rb'))
        X_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/X_test_" + str(i) + ".csv", index_col = 0)
        y_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/y_test_" + str(i) + ".csv", index_col = 0)

        y_pred = model.predict(X_test)
        y_pred = pd.DataFrame(y_pred)
        y_pred.columns = ["y_pred"]
        y_test.columns = ["y_test"]

        df = pd.concat([y_test.reset_index(), y_pred], axis = 1)
        
        multiple_mae = pd.concat([multiple_mae, pd.Series(return_mae(df))], axis = 0)
    
    return multiple_mae

def return_multiple_rho(experiment, model_name):
    
    root = os.getcwd()
    
    multiple_rho = pd.DataFrame()
    
    for i in range(1,6,1):

        filename = root + "/experiment-" + str(experiment) + "/models/" + model_name + "_" + str(i) + ".pkl"

        model = pickle.load(open(filename, 'rb'))
        X_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/X_test_" + str(i) + ".csv", index_col = 0)
        y_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/y_test_" + str(i) + ".csv", index_col = 0)

        y_pred = model.predict(X_test)
        y_pred = pd.DataFrame(y_pred)
        y_pred.columns = ["y_pred"]
        y_test.columns = ["y_test"]

        df = pd.concat([y_test.reset_index(), y_pred], axis = 1)
        
        multiple_rho = pd.concat([multiple_rho, df.groupby("id").apply(return_rho)], axis = 0)
    
    return multiple_rho

def return_multiple_mae(experiment, model_name):
    
    root = os.getcwd()
    
    multiple_mae = pd.DataFrame()
    
    for i in range(1,6,1):

        filename = root + "/experiment-" + str(experiment) + "/models/" + model_name + "_" + str(i) + ".pkl"

        model = pickle.load(open(filename, 'rb'))
        X_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/X_test_" + str(i) + ".csv", index_col = 0)
        y_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/y_test_" + str(i) + ".csv", index_col = 0)

        y_pred = model.predict(X_test)
        y_pred = pd.DataFrame(y_pred)
        y_pred.columns = ["y_pred"]
        y_test.columns = ["y_test"]

        df = pd.concat([y_test.reset_index(), y_pred], axis = 1)
        
        multiple_mae = pd.concat([multiple_mae, df.groupby("id").apply(return_mae)], axis = 0)
    
    return multiple_mae

def return_true_pred_plot(experiment, model_name):
    
    root = os.getcwd()
    
    for i in range(1,6,1):

        filename = root + "/experiment-" + str(experiment) + "/models/" + model_name + "_" + str(i) + ".pkl"

        model = pickle.load(open(filename, 'rb'))
        X_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/X_test_" + str(i) + ".csv", index_col = 0)
        y_test = pd.read_csv(root + "/experiment-" + str(experiment) + "/data/y_test_" + str(i) + ".csv", index_col = 0)

        y_pred = model.predict(X_test)
        y_pred = pd.DataFrame(y_pred)
        y_pred.columns = ["y_pred"]
        y_test.columns = ["y_test"]

        df = pd.concat([y_test.reset_index(), y_pred], axis = 1)
        df = df[df.y_test > 3]
        df = df[df.y_pred > 3]
        df = df[df.y_test < 12]
        df = df[df.y_pred < 12]
        
        plt.scatter(df.y_test, df.y_pred)
        plt.xlim(3, 12)
        plt.ylim(3, 12)
        plt.xlabel("Self-reported sleep duration")
        plt.show("Estimated sleep duation")
        plt.title("Median MAE: " + str(return_multiple_mae(experiment, model_name).median()[0]))
        
def results_single_study(row):
        
    rhos = return_multiple_rho(row[0], row[1][0])
    maes = return_multiple_mae(row[0], row[1][0])
    rhos.columns = ["rho"]
    maes.columns = ["mae"]
    
    results_single_study = pd.DataFrame({"median rho":rhos.median().values[0],
                                         "median mae":maes.median().values[0] * 60,
                                         "min rho":rhos.min().values[0],
                                         "min mae":maes.min().values[0] * 60,
                                         "max rho":rhos.max().values[0],
                                         "max mae":maes.max().values[0] * 60,
                                         "n rho > 0.5":sum(rhos.rho > 0.5)/rhos.shape[0],
                                         "n rho > 0.8":sum(rhos.rho > 0.8)/rhos.shape[0]}, index = [row[1][0]])
    
    return results_single_study

def results_multiple_studies(x):
    
    results_multiple_studies = pd.DataFrame()
    
    for row in x.iterrows():
        
        results_multiple_studies = pd.concat([results_multiple_studies, results_single_study(row)], axis = 0)
        
    return results_multiple_studies

def table_2()
    study_parameters = pd.read_json("study_parameters.json")
    x = pd.DataFrame({"model":["lasso","svr","rf","gbr"]})
    x.index = [1,2,3,4]
    x = results_multiple_studies(x)
    x.to_csv(study_parameters["markdown_path"] + "table_2.csv")