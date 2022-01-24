# Import modules
import pandas as pd
import numpy as np

# Read data
print("Reading the data.")
columns = ["id","application","startTime","endTime"]
log_data = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", usecols = columns)
log_data.drop_duplicates(inplace=True)
log_data.reset_index(inplace=True,drop=True)

def sleep_single_subject(data):
    df = data.copy(deep=True)
    
    try:

        # Set startTime as the index
        df.loc[:,"indexTime"] = pd.to_datetime(df.startTime)
        df.set_index("indexTime", inplace=True)

        # Make startTime and endTime a datetime object
        df.loc[:,"endTime"] = pd.to_datetime(df.endTime).tz_localize(None)
        df.loc[:,"startTime"] = pd.to_datetime(df.startTime).tz_localize(None)

        # Resample the dataframe to minutes
        df = df.groupby('id').resample("min").agg({"startTime":np.min, "endTime":np.max})

        # Forward fill startTime and endTime
        df.startTime = df.startTime.fillna(method="ffill")
        df.endTime = df.endTime.fillna(method="ffill")

        # Reset the index
        df.reset_index(inplace=True)

        # Replace instances where indexTime exceeds startTime by indexTime
        df.startTime[df.indexTime > df.startTime] = df.indexTime

        # Set the index to indexTime again
        df.set_index(["id","indexTime"], inplace = True)

        # Create an all-zero column called binary bins, representing whether a person used their phone or not
        df.loc[:,"binary_bins"] = 0

        # Replace all zeros with a one if startTime is smaller than endTime (meaning a person used their phone)
        df.loc[:,"binary_bins"][df.startTime < df.endTime] = 1

        # Least activity hours
        least_activity = df.reset_index().set_index("indexTime").resample("15Min").sum()
        least_activity.reset_index(inplace=True)
        least_activity["time"] = least_activity.indexTime.dt.time
        least_activity = pd.concat([least_activity.groupby("time").mean(),least_activity.groupby("time").mean()]).rolling(24).mean().dropna().sort_values(by=["binary_bins"])

        stop = pd.to_datetime(least_activity.reset_index().time.astype(str)[0])
        start = stop - pd.Timedelta('6H')

        stop_start = pd.to_datetime(pd.concat([pd.Series(stop), pd.Series(start)])).dt.time

        # Create moving sum of binary states (120 minute time window with epoch at center)
        log_data_moving_sum = df.rolling(120, center = True).sum()

        # Drop generated missingness
        log_data_moving_sum.dropna(inplace=True)

        # Reset the index again
        log_data_moving_sum.reset_index(inplace=True)

        # Add edges of the time window
        log_data_moving_sum["startTime"] = log_data_moving_sum["indexTime"] - pd.Timedelta('60Min')
        log_data_moving_sum["endTime"] = log_data_moving_sum["indexTime"] + pd.Timedelta('60Min')

        # Select bins with subthreshold smartphone usage
        selected_bins = log_data_moving_sum[log_data_moving_sum["binary_bins"] < 2].dropna()

        # Create a date variable so we can group according to date
        selected_bins.loc[:,"date"] = pd.to_datetime(selected_bins["indexTime"]).dt.date.astype(str)

        # Select hour variable so we can deselect hours outside rest range
        selected_bins.loc[:,"hour"] = pd.to_datetime(selected_bins["indexTime"]).dt.hour

        # Select startTime/endTime combinations that overlap with stop/start
        # If either the startTime or the endTime is between the regular sleep time, then we assume the person is sleeping
        starttime = selected_bins["startTime"].dt.time.between(stop_start.iloc[1], stop_start.iloc[0])
        endtime = selected_bins["endTime"].dt.time.between(stop_start.iloc[1], stop_start.iloc[0])

        # We select rows where we believe people are sleeping
        selected_bins = selected_bins[(starttime) | (endtime)]

        # We then get per date the minimum startTime and maximal endTime
        sleep_onset = selected_bins.groupby("date").startTime.min()
        sleep_offset = selected_bins.groupby("date").endTime.max()

        sleep = pd.merge(sleep_onset, sleep_offset, on = "date", how = "outer")
        sleep["duration"] = sleep["endTime"] - sleep["startTime"]

        return sleep
    
    except:
        
        pass

sleep_duration = log_data.groupby("id").apply(sleep_single_subject)

sleep_duration.reset_index(inplace=True) 
sleep_duration.columns = ["id","date","sleep_onset","wake_time","sleep_duration"]
sleep_duration.to_csv("naive-baseline-features.csv")