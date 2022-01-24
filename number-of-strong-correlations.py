import pandas as pd

def number_of_strong_correlations(df):
    return (df >= 0.8).sum()

def number_of_medium_correlations(df):
    return (df >= 0.5).sum()

correlation_strength = pd.DataFrame()

print("Nomothetic")
for mod in ["RF","SVR","XGBoost"]:
    for n in range(3):
        df = pd.read_csv("/home/haalbers/sleep-estimation/gen/paper/nomothetic-correlations" + mod + "_FEATURE_SET_" + str(n) + ".csv", index_col = 0)
        print("Strong correlations")
        print(number_of_strong_correlations(df))
        print("Medium correlations")
        print(number_of_medium_correlations(df))
        
        correlation_strength = pd.concat([correlation_strength, 
                                          pd.DataFrame({"model":mod,
                                                        "feature_set":n,
                                                        "strong_correlations":number_of_strong_correlations(df)[0],
                                                        "medium_correlations":number_of_medium_correlations(df)[0]}, index = ["nomothetic"])])

print("Idiographic") 
for mod in ["rf","svr","xg_boost"]:
    for n in range(3):
        df = pd.read_csv("/home/haalbers/sleep-estimation/gen/paper/idiographic-correlations" + mod + "_FEATURE_SET_" + str(n) + ".csv", index_col = 0)
        print("Strong correlations")
        print(number_of_strong_correlations(df))
        print("Medium correlations")
        print(number_of_medium_correlations(df))
                                          
        correlation_strength = pd.concat([correlation_strength, 
                                  pd.DataFrame({"model":mod,
                                                "feature_set":n,
                                                "strong_correlations":number_of_strong_correlations(df)[0],
                                                "medium_correlations":number_of_medium_correlations(df)[0]}, index = ["idiographic"])])

print("Baseline")
        
df = pd.read_csv("/home/haalbers/sleep-estimation/src/naive-baseline-correlations.csv", index_col = 0)
print("Strong correlations")
print(number_of_strong_correlations(df.test_correlation))
print("Medium correlations")
print(number_of_medium_correlations(df.test_correlation))

correlation_strength = pd.concat([correlation_strength, 
                                  pd.DataFrame({"model":mod,
                                                "feature_set":n,
                                                "strong_correlations":number_of_strong_correlations(df.test_correlation),
                                                "medium_correlations":number_of_medium_correlations(df.test_correlation)}, index = ["naive-baseline"])])
                                  
correlation_strength.to_csv("/home/haalbers/sleep-estimation/gen/paper/correlation-strength.csv")
