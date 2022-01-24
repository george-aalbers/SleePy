# Import modules
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv("/home/haalbers/sleep-estimation/src/naive-baseline.csv", usecols=['id','sleep_duration','reported_sleep_duration'])

def correlations_test_set_single_subject(df):
    n = df.shape[0]-(df.shape[0]//5)
    train_set = df.iloc[:n,1:]
    test_set = df.iloc[n:,1:]
    train_correlation = spearmanr(train_set.sleep_duration, train_set.reported_sleep_duration)
    test_correlation = spearmanr(test_set.sleep_duration, test_set.reported_sleep_duration)
    results = pd.DataFrame({"train_correlation": train_correlation[0], 
                            "train_p-value": train_correlation[1],
                            "test_correlation": test_correlation[0], 
                            "test_p-value": test_correlation[1], 
                            "train_n":train_set.shape[0],
                            "test_n":test_set.shape[0]}, 
                           index=[df.id.unique()])
    return results

results = df.groupby("id").apply(correlations_test_set_single_subject).dropna()
results.to_csv("/home/haalbers/sleep-estimation/src/naive-baseline-correlations.csv") 