# Print message
print("Starting data preparation for personalized sleep estimation.")

# Import modules
import pandas as pd
import numpy as np

# Read data
print("Reading the data.")
log_data = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", index_col = 0)

# Get startMinute and endMinute
print("Getting and recoding timestamps")
log_data["startMinute"] = pd.to_datetime(log_data.dt_s).dt.minute
log_data["endMinute"] = pd.to_datetime(log_data.dt_e).dt.minute

# Replace those with 1 (first half) or 2 (second half)
dictionary = dict(zip(list(np.arange(60)),
                      list(np.tile(np.concatenate([np.repeat(1,7), np.repeat(2,8)]), 4))))
log_data["startMinute"] = log_data["startMinute"].replace(dictionary)
log_data["endMinute"] = log_data["endMinute"].replace(dictionary)

# Resample log_data to quarter-of-an-hourly timescale 
print("Resampling to different timescale.")
log_data.reset_index(inplace=True)
log_data["time"] = pd.to_datetime(log_data["time"])
log_data.set_index("time", inplace=True)
log_data = log_data.groupby("id").resample("15Min").agg({"durationSeconds":np.sum})
log_data.fillna(0, inplace=True)
log_data.reset_index(inplace=True)

# Write to file
log_data.to_csv("resampled-log-data.csv")