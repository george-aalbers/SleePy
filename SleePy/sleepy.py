import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def binarize(data):
    
    # Make deep copy of the data
    df = data.copy(deep=True)

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

    return df

def quarterize(df):
    df = df.reset_index().set_index('indexTime').groupby('id').resample('15Min').sum().reset_index()
    df.columns = ['id', 'time', 'duration']
    df['date'] = pd.to_datetime(df['time']).dt.date
    df['time'] =  pd.to_datetime(df['time']).dt.time
    return df

def pivot_data(df):

    # Pivot the table so that we get a column for each quarter-of-an-hour per day
    log_data_pivoted = df.pivot_table(columns = ["time"], values = ["duration"], index = ["id","date"]).iloc[:,1:].reset_index()

    # Forward shift all rows and merge with original dataframe and drop missings
    log_data_pivoted_shifted = pd.merge(log_data_pivoted.shift(1), log_data_pivoted, on = ["id","date"], how = "outer").dropna()

    # Drop multi-index level
    log_data_pivoted_shifted = log_data_pivoted_shifted.droplevel(axis=1, level=0)

    # Rename columns
    log_data_pivoted_shifted.columns.values[0] = "id"
    log_data_pivoted_shifted.columns.values[1] = "date"
    
    # Set index
    log_data_pivoted_shifted.set_index(['id','date'], inplace = True)
    
    # Divide by 15 to bring values within range 0 to 1
    log_data_pivoted_shifted = log_data_pivoted_shifted/15
    
    return log_data_pivoted_shifted

def get_data(data):
    
    # Downsample to minutes
    df = binarize(data)
    
    # Downsample to quarters-of-an-hour
    df = quarterize(df)
    
    # Pivot the dataframe to get long format
    df = pivot_data(df)
    
    # Return the relevant columns
    return df.iloc[:,83:-43]

def load_model(model_name = "model.pkl"):
    
    # Load and return trained random forest 
    model = pickle.load(open(model_name, 'rb'))
    
    return model

def estimate_sleep(data):
    
    '''
    This function estimates sleep duration from smartphone application usage log data. It outputs a dataframe with an estimated sleep duration for each date per participant. 
    
    The function was developed on data logged by MobileDNA, but can also be used for other providers. To function, it does require the data to have the following three columns: 
    (1) 'id' - the participant ID, 
    (2) 'startTime' - a timestamp for the start of an application event,
    (3) 'endTime' - a timestamp for the end of an application event.
    
    '''
    
    # Get application usage log data in the right format
    df = get_data(data)
    
    # Load trained model
    model = load_model()
    
    # Use trained model (e.g., random forest) to make predictions
    predictions = model.predict(df)
    
    # Make dataframe with predictions
    predictions = pd.DataFrame(predictions, index = df.index)
    predictions.reset_index(inplace = True)
    predictions.columns = ['id', 'date', 'sleep duration']
    
    # Return predictions
    return predictions