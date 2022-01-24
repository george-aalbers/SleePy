# Import modules
import pandas as pd
import numpy as np

# Read data
print("Reading the data for pivot")
data = pd.read_csv("merged-data.csv", index_col = 0)

# Get experience sampling data from dataframe
esm_data = data[['id',
                 'date',
                 'sleep_latency',
                 'sleep_inertia',
                 'sleep_quality',
                 'bed_time',
                 'sleep_time',
                 'wake_time',
                 'self_reported_sleep_duration']]

# Get time for log data from dataframe
data["time_x"] = pd.to_datetime(data.time).dt.time

# Pivot the table so that we get a column for each quarter-of-an-hour per day
print("Pivoting the dataframe.")

# We do this for the different features we include in our model
for value in ["durationSeconds","startMinute","endMinute"]:

    data_pivot = data.pivot(columns = ["time_x"], values = value, index =
                            ["id","date"]).iloc[:,1:].reset_index()

    # Forward shift all rows and merge with original dataframe and drop missings
    print("Shifting forward and merging with non-shifted data.")
    new_df = pd.merge(data_pivot.shift(1), data_pivot, on = ["id","date"], how = "outer").dropna()

    # Then merge with experience sampling data and drop missings
    new_df = pd.merge(esm_data, new_df, on = ["id","date"], how = "outer").dropna()

    # Write to file
    print("Writing the preprocessed data to file.")
    new_df.to_csv("data-" + value + ".csv")