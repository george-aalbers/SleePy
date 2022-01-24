import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection as fdr
import pandas as pd
import numpy as np 

def spearman(df):
    return spearmanr(df.y_pred, df.y_test)[1]

for model in ["rf", "xg","svr"]:

    if model == "xg":
        mod = "XGBoost"
    elif model == "svr":
        mod = "SVR"
    else:
        mod = "RF"
    
    for feature_set in range(3):

        df = pd.DataFrame()
        
        if feature_set == 0:
            features = "(duration only)"
        elif feature_set == 1:
            features = "(timestamps only)"
        else:
            features = "(both duration and timestamps)"
        
        for cv in range(5):

            data = pd.read_csv("/home/haalbers/sleep-estimation/gen/analysis/output/nomothetic/predictions/" + model + "-self_reported_sleep_duration_CV_" + str(cv) + "_FEATURE_SET_" + str(feature_set) + ".csv", index_col = 0)
            df = pd.concat([df, data], axis = 0)
            
        df.groupby("id").apply(spearman).to_csv("/home/haalbers/sleep-estimation/gen/paper/nomothetic-correlations" + mod + "_FEATURE_SET_" + str(feature_set) + ".csv")
  
results = pd.DataFrame()
for mod in ["RF", "XGBoost","SVR"]:
    for feature_set in range(3):
        df = pd.read_csv("/home/haalbers/sleep-estimation/gen/paper/nomothetic-correlations" + mod + "_FEATURE_SET_" + str(feature_set) + ".csv")
        median_correlation = pd.DataFrame({"mod":mod, 
                                           "feature_set":feature_set, 
                                           "median_corr": df.iloc[:,1].median(),
                                           "mini_corr": df.iloc[:,1].min(),
                                           "maxi_corr": df.iloc[:,1].max()}, index = [mod])
        results = pd.concat([results, median_correlation], axis = 0)
        results.to_csv("/home/haalbers/sleep-estimation/gen/paper/nomothetic-results.csv")