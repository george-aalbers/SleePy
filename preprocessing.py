# Print message
print("Starting preprocessing for sleep estimation study.")

# Import modules
import pandas as pd
import numpy as np

# Functions for preprocessing of the data

# Imports
import pandas as pd
import numpy as np

def scale_features():    
    X = pd.read_csv("X.csv", index_col = 0)
    X = X/900
    X.to_csv("X.csv")

def order_features():
    X = pd.read_csv("X.csv", index_col = 0)
    X = X[X.columns.sort_values(ascending=False).tolist()]
    yesterday = X.columns[:12].tolist()
    today = X.columns[12:].tolist()
    today.extend(yesterday)
    X = X[today]
    X.to_csv("X.csv")
    
def transform_to_dataframe(df):
    return pd.DataFrame(df)
    
def select_features_targets_ids(study_parameters):
    
    # Read preprocessed data
    data = pd.read_csv("data.csv", index_col = 0, low_memory = False)
    
    # Split into X and y
    ids = data[study_parameters["id_variable"]]
    X = data[study_parameters["features"]]
    y = data[study_parameters["targets"]]
    
    # Set X and y index to ids
    X.index = ids
    y.index = ids
    
    # Write to file
    ids.to_csv("ids.csv")
    X.to_csv("X.csv")
    y.to_csv("y.csv")
    
def center_all_data_single_subject(df, study_parameters):
    
    # Center all data for one participant
    df.iloc[:,1:] = df.iloc[:,1:] - df.iloc[:,1:].mean()
    
    return df
    
def within_person_center_nomothetic(train, test, study_parameters):
    
    # Transform data to dataframe
    train = transform_to_dataframe(train).reset_index()
    test = transform_to_dataframe(test).reset_index()
    
    # Center all data for multiple participants
    train = train.groupby("id").apply(center_all_data_single_subject, study_parameters)
    test = test.groupby("id").apply(center_all_data_single_subject, study_parameters)
    
    # Set index to id variable
    train.set_index("id", inplace=True)
    test.set_index("id", inplace=True)
    
    return train, test
    
def within_person_center_idiographic(train, test, study_parameters):
    
    # Center one person's data based on mean in train data
    
    # Get mean in train data
    mean = train.mean()
    
    # Subtract from train data
    train = train - mean
    
    # And subtract from test data
    test = test - mean
    
    return train, test
        
def center(train, test, study_parameters):
    
    # Center data for each person, method depends on experiment type
    if study_parameters["experiment_type"] == "idiographic":
        return within_person_center_idiographic(train, test, study_parameters)
    elif study_parameters["experiment_type"] == "nomothetic":
        return within_person_center_nomothetic(train, test, study_parameters)

def flatten_targets(df):
    return np.ravel(df)

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
    esm_data = esm_data[['id', 'date', 'self_reported_sleep_duration']]
    return esm_data

def merge_data(log_data, esm_data):
    data = pd.merge(log_data, esm_data, on = ["id","date"], how = "outer")
    return data
    
def remove_duplicates(data):
    return data[~data[["id","date","time"]].duplicated()]

def preprocess():

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

    # Drop days without any smartphone use
    data = pd.read_csv("data.csv", index_col = 0)
    print("Before dropping faulty features:", data.shape)
    data = data[data.iloc[:,3:].sum(axis=1) != 0]
    print("After dropping faulty features:", data.shape)
    
    # Drop participants with < 50 observations
    pps = data.id.value_counts()[data.id.value_counts() > 49]
    data = data[data.id.isin(pps.index.tolist())]
    data.reset_index(inplace=True,drop=True)
    data.to_csv("data.csv")

    print("Finished data preprocessing for sleep estimation study.")