# Import modules
import pandas as pd
import numpy as np

# Read data
print("Reading the data.")
log_data = pd.read_csv("resampled-log-data.csv", index_col = 0)
esm_data = pd.read_csv("/home/haalbers/dissertation/experience-sampling-clean.csv", index_col = 0)

# Resample esm_data to daily timescale
esm_data.rename({"Name":"id"}, axis = 1, inplace=True)
esm_data["time_esm"] = pd.to_datetime(esm_data["Response Time_ESM_day"], errors="coerce", utc=True)
esm_data = esm_data[~pd.isnull(esm_data.time_esm)]
esm_data.set_index("time_esm", inplace=True)
esm_data = esm_data.groupby("id").resample("D").mean()
esm_data.reset_index(inplace=True)

# Choose relevant ESM data and merge with log data
esm_data = esm_data[['id',
                     'time_esm',
                     'sleep_latency',
                     'sleep_inertia',
                     'sleep_quality',
                     'bed_time',
                     'sleep_time',
                     'wake_time',
                     'self_reported_sleep_duration']]

# Create date column to enable merge
esm_data["date"] = pd.to_datetime(esm_data.time_esm).dt.date
log_data["date"] = pd.to_datetime(log_data.time).dt.date

# Do an outer merge on id and date
print("Merging log and ESM data.")
data = pd.merge(log_data, esm_data, on = ["id","date"], how = "outer")

# Remove duplicates
print("Removing duplicates.")
data = data[~data[["id","date","time"]].duplicated()]

# Write to file
data.to_csv("merged-data.csv")