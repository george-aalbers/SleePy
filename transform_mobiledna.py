print("LOADING AND TRANSFORMING MOBILEDNA")

import pandas as pd
import os
import numpy as np

def load_transform_mobile_dna(separate_files = True, log_data_path = "path_to_log_data"):

    '''
    parameter separate_files: a boolean indicating whether we have separate csvs for each participant (True) or not (False).
    parameter log_data_path: a string indicating the log data path (either the folder containing separate files or the csv with all participants)
    '''
        
    if separate_files:
        files = os.listdir(log_data_path)
        N = 0
        df = pd.DataFrame()
        for file in files:
            try: 
                df_ss = pd.read_csv(log_data_path + file, sep = ";", index_col = 0)
                
                df_adj = pd.DataFrame()
                df_ss["dt_s"] = pd.to_datetime(df_ss["startTime"])
                df_ss["dt_e"] = pd.to_datetime(df_ss["endTime"])
                for i, row in df_ss.iterrows():
                    ts_start = row["dt_s"]
                    ts_end = row["dt_e"]
                    x = pd.to_datetime(ts_start.round(freq='15T'))

                    if x < ts_start:
                        x = pd.to_datetime(ts_start.round(freq='15T') + pd.Timedelta(minutes=15))
                    else:
                        x = pd.to_datetime(ts_start.round(freq='15T'))

                    ts_new = [ts_start, x]

                    while x < ts_end:
                        x = x + pd.Timedelta(minutes=15)
                        ts_new.append(x)
                    ts_new[-1] = ts_end
                    new_rows = pd.DataFrame(np.repeat([row], len(ts_new)-1, axis = 0))
                    new_rows.columns = df_ss.columns
                    for j in range(new_rows.shape[0]):
                        timestamps = ts_new[(j):(j+2)]
                        new_rows.loc[j,["dt_s","dt_e"]] = timestamps
                    df_adj = pd.concat([df_adj, new_rows])
                  
                df = pd.concat([df, df_adj], axis = 0)
                N += 1
                print("Processed", N, "participants.")
            except:
                print("Could not merge", file)
                
    else:

        df = pd.read_csv(log_data_path, sep = ";", index_col = 0)

    return df

# Load mobileDNA data and transform to 15 minute splits
log_data_path = "/home/haalbers/dissertation/data/mobile-dna/"
df = load_transform_mobile_dna(separate_files = True, log_data_path = log_data_path)
df.to_csv("mobiledna-transformed.csv")
print("----------------------------------")