import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

for mod in ["RF", "XGBoost", "SVR"]:
    for num in range(3):
        if num == 0:
            feature_set = " (duration only)"
        elif num == 1:
            feature_set = " (timestamps only)"
        else:
            feature_set = " (both duration and timestamps)"
        df = pd.read_csv("/home/haalbers/sleep-estimation/gen/paper/nomothetic-correlations"+mod+"_FEATURE_SET_"+str(num)+".csv", index_col = 0)
        sns.histplot(df.iloc[:,0].values)
        plt.xlabel("Person-specific Spearman rank-order correlation")
        plt.ylabel("Number of participants")
        plt.xlim(0,1)
        plt.ylim(0,150)
        
        if mod == "XGBoost":
            plt.title("GB" + feature_set)
            plt.savefig("/home/haalbers/sleep-estimation/gen/paper/figure-nomothetic-correlations-xgboost-" + feature_set + ".png")
            plt.clf()
        else:
            plt.title(mod.upper() + feature_set)
            plt.savefig("/home/haalbers/sleep-estimation/gen/paper/figure-nomothetic-correlations-" + mod + "-" + feature_set + ".png")
            plt.clf()