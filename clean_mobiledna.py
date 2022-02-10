print("CLEANING MOBILEDNA")

######################################################################
# Load packages, data, and remove participants who dropped out early #
######################################################################

import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import datetime

# Load data and print number of total IDs in the dataset
df1 = pd.read_csv("mobiledna-transformed.csv", index_col = 0)
print("In total, we have", len(df1.id.unique().tolist()), "participants in our mobileDNA dataset")

# Check how many we have in our key and print how many participants we included
key = pd.read_csv("anonymized_key.csv", index_col = 0)
print("A total of", df1[df1.id.isin(key.MobileDNA)].id.nunique(), "decided to participate.")

# Drop any non-participating IDs
data = df1[df1.id.isin(key.MobileDNA)]

# Calculate duration of application events
duration = (pd.to_datetime(data["dt_e"]) - pd.to_datetime(data["dt_s"]))

# Transform duration (in ns) to seconds and add to df
data.loc[:,"durationSeconds"] = duration.astype(int)/(1000000000) 

# Remove duplicates
data.drop_duplicates(subset=["id", "dt_s", "dt_e"], inplace = True)
data.drop_duplicates(subset=["id", "dt_s"], inplace = True)

# Reset index
data.reset_index(inplace = True, drop = True)

# Add hour and date to dataframe
data.loc[:,"hour"] = pd.to_datetime(data["dt_s"]).dt.hour
data.loc[:,"date"] = pd.to_datetime(data["dt_s"]).dt.date

# Sort dataframe on ID and startTime of application event
data = data.sort_values(["id", "dt_s"])

# Check for overlapping time intervals (when endTime of previous app > startTime of current app)
for i in data.id.unique().tolist():
    dfp = data[data["id"] == i]

    dfp_shift = pd.concat([dfp.dt_s, dfp.dt_e.shift(1)], axis = 1)
    dfp_shift.dropna(inplace = True)

    dfp_shift.loc[:,"dt_s"] = pd.to_datetime(dfp_shift["dt_s"])
    dfp_shift.loc[:,"dt_e"] = pd.to_datetime(dfp_shift["dt_e"])
    
    n_problems = dfp_shift[(dfp_shift.dt_s - dfp_shift.dt_e) < datetime.timedelta(minutes=0)].shape[0]

    if n_problems > 0:    
        print(i)
        print(n_problems)
        
# This is mostly an issue for one user, who will be removed from the dataset.
data = data[~data.id.isin(["ca9bc5f3-c337-4abd-aa99-8742adf21ac5"])]

# Transform dataset to hourly data and check for impossible values (i.e., > 3600 seconds)
df_hour = data.groupby(["id", "date", "hour"]).agg({'durationSeconds': np.sum})
print(df_hour[df_hour.durationSeconds > 3600])

# Remove some superfluous columns
data.index = pd.to_datetime(data["dt_s"])
del data["session"]
del data["notificationId"]
del data["data_version"]

################
# Rename users #
################

# Transform key to dictionary
mobiledna_id = key.iloc[:,1].values.tolist()
ethica_id = key.iloc[:,0].values.tolist()
key = dict(zip(mobiledna_id, ethica_id))
data.replace(key, inplace=True)

# Do an additional clean and write to file
data = data[~data.duplicated(subset=["id","startTime","endTime"])]
duration = pd.to_datetime(data.dt_e) - pd.to_datetime(data.dt_s)
duration = duration.astype("int64")/1000000000
data.loc[:,"duration"] = duration

# Create a new "faketime" series to determine in which quarter of an hour the observation falls
# We can use this variable if we want to use pandas to aggregate data per quarter-of-an-hour.
data.dt_s = pd.to_datetime(data.dt_s)
data.loc[:,"quarter"] = data.dt_s.dt.minute
data.loc[:,"quarter"][(data.dt_s.dt.minute > 0) & (data.dt_s.dt.minute < 15)] = 0
data.loc[:,"quarter"][(data.dt_s.dt.minute >= 15) & (data.dt_s.dt.minute < 30)] = 1
data.loc[:,"quarter"][(data.dt_s.dt.minute >= 30) & (data.dt_s.dt.minute < 45)] = 2
data.loc[:,"quarter"][(data.dt_s.dt.minute >= 45)] = 3
data.loc[:,"hour"] = data.dt_s.dt.hour

# Set this "faketime" series as the index
data.set_index(pd.to_datetime(data["date"].astype(str) + " " + data["hour"].astype(str) + ":" + (data["quarter"]*15).astype(str) + ":00"), inplace=True)
data.reset_index(inplace=True)
data.rename({"index":"time"}, axis = 1, inplace=True)
data.to_csv("mobiledna-clean.csv")
print("----------------------------------")