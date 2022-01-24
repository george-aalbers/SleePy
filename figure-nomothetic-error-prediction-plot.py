import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
import pandas as pd
import numpy as np 

for model in ["rf", "xg","svr"]:

    if model == "xg":
        mod = "GB"
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
            
        sns.set_theme()
        sns.scatterplot(
            data=data,
            x="y_test",
            y="y_pred", 
            color="k"
        )
        plt.xlabel("Self-reported sleep duration")
        plt.ylabel("Estimated sleep duration")
        plt.xlim(2,12)
        plt.ylim(2,12)
        plt.text(x = 2.5, y = 11, s = "Spearman correlation: " + str(np.round(spearmanr(df.y_pred, df.y_test)[0],2)) + ", p-value < .001")
        plt.title("Performance of " + mod + " " + str(features))
        sns.kdeplot(
            data=data,
            x="y_test",
            y="y_pred",
            levels=5,
            fill=True,
            alpha=0.6,
            cut=2
        )
        plt.savefig("/home/haalbers/sleep-estimation/gen/paper/figure-nomothetic-prediction-error-" + "-" + model + "-" + str(feature_set) + ".png")
        plt.clf()