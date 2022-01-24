# Print message
print("Starting data preparation for personalized sleep estimation.")

# Import modules
import pandas as pd
import numpy as np

def resample_data(log_data, esm_data):
    
    # Resample log_data to quarter-of-an-hourly timescale 
    print("Resampling to different timescale.")
    log_data.reset_index(inplace=True)
    log_data["time"] = pd.to_datetime(log_data["time"])
    log_data.set_index("time", inplace=True)
    log_data = log_data.groupby("id").resample("15Min").agg({"durationSeconds":np.sum})
    log_data.reset_index(inplace=True)
    
    # Create date column to enable merge
    log_data["date"] = pd.to_datetime(log_data.time.dt.date)

    # Resample esm_data to daily timescale
    esm_data.rename({"Name":"id"}, axis = 1, inplace=True)
    esm_data["date"] = pd.to_datetime(esm_data["Response Time_ESM_day"].str[:10])
    esm_data.set_index("date", inplace=True)
    esm_data = esm_data.groupby("id").resample("D").mean()
    esm_data.reset_index(inplace=True)
    
    return log_data, esm_data

def select_columns(esm_data):
    esm_data = esm_data[['id',
                     'date',
                     'sleep_latency',
                     'sleep_inertia',
                     'sleep_quality',
                     'bed_time',
                     'sleep_time',
                     'wake_time',
                     'self_reported_sleep_duration']]
    return esm_data

def merge_data(log_data, esm_data):
    data = pd.merge(log_data, esm_data, on = ["id","date"], how = "outer")
    return data
    
def remove_duplicates(data):
    return data[~data[["id","date","time"]].duplicated()]

# Read data
print("Reading the data.")
log_data = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", index_col = 0)
esm_data = pd.read_csv("/home/haalbers/dissertation/experience-sampling-clean.csv", index_col = 0)

# Resample data
log_data, esm_data = resample_data(log_data, esm_data)

# Choose relevant ESM data and merge with log data
esm_data = select_columns(esm_data)

# Remove duplicates
print("Removing duplicates.")
log_data = remove_duplicates(log_data)

# Get time variable
log_data["time"] = pd.to_datetime(log_data.time).dt.time
log_data = log_data[~pd.isnull(log_data["time"])]

# Pivot the table so that we get a column for each quarter-of-an-hour per day
print("Pivoting the dataframe.")
log_data_pivoted = log_data.pivot_table(columns = ["time"], values = ["durationSeconds"], index = ["id","date"]).iloc[:,1:].reset_index()

# Forward shift all rows and merge with original dataframe and drop missings
print("Shifting forward and merging with non-shifted data.")
log_data_pivoted_shifted = pd.merge(log_data_pivoted.shift(1), log_data_pivoted, on = ["id","date"], how = "outer").dropna()

# Drop multi-index level
log_data_pivoted_shifted = log_data_pivoted_shifted.droplevel(axis=1, level=0)

# Rename columns
log_data_pivoted_shifted.columns.values[0] = "id"
log_data_pivoted_shifted.columns.values[1] = "date"

# Then merge with experience sampling data and drop missings
data = pd.merge(esm_data, log_data_pivoted_shifted, on = ["id","date"], how = "outer").dropna()

# Write to file
print("Writing the preprocessed data to file.")
data.to_csv("data.csv")

print("Finished data preparation for personalized sleep estimation.")