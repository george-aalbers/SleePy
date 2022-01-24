# Print message
print("Starting data preparation for personalized sleep estimation.")

# Import modules
import pandas as pd
import numpy as np

# Read data
print("Reading the data.")
log_data = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", index_col = 0)
esm_data = pd.read_csv("/home/haalbers/dissertation/experience-sampling-clean.csv", index_col = 0)

# Resample log_data to quarter-of-an-hourly timescale 
print("Resampling to different timescale.")
log_data.reset_index(inplace=True)
log_data["time"] = pd.to_datetime(log_data["time"])
log_data.set_index("time", inplace=True)
log_data = log_data.groupby("id").resample("15Min").agg({"durationSeconds":np.sum})
log_data.reset_index(inplace=True)

# Resample esm_data to daily timescale
esm_data.rename({"Name":"id"}, axis = 1, inplace=True)
esm_data["dt"] = pd.to_datetime(esm_data["rounded_time"], errors="coerce", utc=True)
esm_data = esm_data[~pd.isnull(esm_data.dt)]
esm_data.set_index("dt", inplace=True)
esm_data = esm_data.groupby("id").resample("D").mean()
esm_data.reset_index(inplace=True)

# Choose relevant ESM data and merge with log data
esm_data = esm_data[['id',
                     'dt',
                     'sleep_latency',
                     'sleep_inertia',
                     'sleep_quality',
                     'bed_time',
                     'sleep_time',
                     'wake_time',
                     'self_reported_sleep_duration']]

# Create date column to enable merge
esm_data["date"] = esm_data.dt.dt.date
log_data["date"] = log_data.time.dt.date

# Do an outer merge on id and date
print("Merging log and ESM data.")
data = pd.merge(log_data, esm_data, on = ["id","date"], how = "outer")

# Remove duplicates
print("Removing duplicates.")
data = data[~data[["id","date","time"]].duplicated()]

# Get time variable
data["time_x"] = pd.to_datetime(data.time).dt.time

# Pivot the table so that we get a column for each quarter-of-an-hour per day
print("Pivoting the dataframe.")
data_pivot = data.pivot(columns = ["time_x"], values = ["durationSeconds"], index = ["id","date"]).iloc[:,1:].reset_index()

# Forward shift all rows and merge with original dataframe and drop missings
print("Shifting forward and merging with non-shifted data.")
data = pd.merge(data_pivot.shift(1), data_pivot, on = ["id","date"], how = "outer").dropna()

# The merge with experience sampling data and drop missings
data = pd.merge(esm_data, data, on = ["id","date"], how = "outer").dropna()

# Write to file
print("Writing the preprocessed data to file.")
data.to_csv("/home/haalbers/sleep-estimation/gen/data-preparation/output/data.csv")
data.to_csv("/home/haalbers/sleep-estimation/gen/analysis/input/data.csv")

print("Finished data preparation for personalized sleep estimation.")