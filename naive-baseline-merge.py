import pandas as pd

log = pd.read_csv("naive-baseline-features.csv", index_col = 0)

esm = pd.read_csv("/home/haalbers/dissertation/experience-sampling-clean.csv", usecols = ['id', 'Response Time_ESM_day', 'sleep_time', 'wake_time', 'self_reported_sleep_duration'])
esm.columns = ["id","time","sleep_time","wake_time","reported_sleep_duration"]
esm["date"] = pd.to_datetime(esm.time.str[:10])
esm = esm.groupby(["id","date"]).mean()
esm.reset_index(inplace=True)

esm = esm[esm.reported_sleep_duration > 3]
esm = esm[esm.reported_sleep_duration < 13]

log["date"] = pd.to_datetime(log["date"])
log["sleep_duration"] = pd.to_datetime(log["sleep_duration"].str[7:]).dt.hour + pd.to_datetime(log["sleep_duration"].str[7:]).dt.minute/60 

merged_df = pd.merge(esm, log, on = ["id","date"], how = "outer")
merged_df.dropna(inplace=True)
merged_df.to_csv("naive-baseline.csv")