# Import modules
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv("/home/haalbers/sleep-estimation/src/naive-baseline.csv", usecols=['id','sleep_duration','reported_sleep_duration'])

def select_test_data_single_subject(df):
    n = df.shape[0]-(df.shape[0]//5)
    test_set = df.iloc[n:,1:]
    return test_set

def plot_prediction_error(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    data = df.groupby("id").apply(select_test_data_single_subject)
    
    sns.set_theme()
    sns.scatterplot(
        data=data,
        x="reported_sleep_duration",
        y="sleep_duration", 
        color="k"
    )
    plt.xlabel("Self-reported sleep duration")
    plt.ylabel("Estimated sleep duration")
    plt.xlim(2,12)
    plt.ylim(2,12)
    plt.text(x = 2.5, y = 11, s = "Spearman correlation: " + str(np.round(spearmanr(df.sleep_duration, df.reported_sleep_duration)[0],2)) + ", p-value < .001")
    plt.title("Performance of naive baseline model")
    sns.kdeplot(
        data=data,
        x="reported_sleep_duration",
        y="sleep_duration",
        levels=5,
        fill=True,
        alpha=0.6,
        cut=2
    )
    plt.savefig("/home/haalbers/sleep-estimation/gen/paper/naive-baseline-prediction-error-plot.png")
    plt.clf()

plot_prediction_error(df)