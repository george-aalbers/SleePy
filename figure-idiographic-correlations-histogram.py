import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

for mod in ["rf", "xg_boost", "svr"]:
    for num in range(3):
        if num == 0:
            feature_set = " (duration only)"
        elif num == 1:
            feature_set = " (timestamps only)"
        else:
            feature_set = " (both duration and timestamps)"
        df = pd.read_csv("/home/haalbers/sleep-estimation/gen/paper/idiographic-correlations"+mod+"_FEATURE_SET_"+str(num)+".csv", index_col = 0)
        sns.histplot(df.iloc[:,0].values)
        plt.xlabel("Person-specific Spearman rank-order correlation")
        plt.ylabel("Number of participants")
        plt.ylim(0,45)
        
        if mod == "xg_boost":
            plt.title("GB" + feature_set)
            plt.savefig("/home/haalbers/sleep-estimation/gen/paper/figure-idiographic-correlations-xgboost-" + feature_set + ".png")
            plt.clf()
        else:
            plt.title(mod.upper() + feature_set)
            plt.savefig("/home/haalbers/sleep-estimation/gen/paper/figure-idiographic-correlations-" + mod + "-" + feature_set + ".png")
            plt.clf()