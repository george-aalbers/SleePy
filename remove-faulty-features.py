import pandas as pd
data = pd.read_csv("data.csv", index_col = 0)
features = pd.read_csv("features.csv", index_col = 0)
features = sum(features.values.tolist(), [])
print("Before dropping faulty features:", data.shape)
data = data[data[features].sum(axis=1) != 0]
print("After dropping faulty features:", data.shape)
data.to_csv("data.csv")