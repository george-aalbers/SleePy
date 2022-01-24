def select_features():

    '''
    Description: Function that merges subsets of features with targets    
    '''
    
    import pandas as pd
    features_merged = pd.DataFrame()

    for file in ["data-durationSeconds.csv", "data-startMinute.csv", "data-endMinute.csv"]:

        # Read file
        df = pd.read_csv(file, index_col = 0)
        
        # Get targets from file
        targets = df.iloc[:,:9]
        
        # Aggregate per person per date
        df = df.groupby(['id','date']).mean().reset_index()
        
        # Get column names for features
        feature_names = df.columns.tolist()[93:-48]
        feature_names.extend(["id","date"])
        
        # Select features (and id and date)
        features = df[feature_names]
        
        # Set index to id and date
        features.set_index(["id","date"], inplace=True)
        
        # Modify feature names
        names = file[5:-4] + "_" + pd.Series(df.columns.tolist()[93:-48])
        names = names.str.replace("_x","")
        names = names.str.replace("_y","")
        
        # Change column names for feature dataframe
        features.columns = names
        features_merged = pd.concat([features_merged, features], axis = 1)

    # Reset index from feature dataframe
    features_merged.reset_index(inplace=True)
    
    # Merge targets with features
    data = pd.merge(targets, features_merged, on = ["id","date"], how = "inner")

    # Aggregate by id and date
    data = data.groupby(["id","date"]).mean().reset_index()
    
    # Get names for features and targets
    features = pd.Series(data.columns[9:])
    targets = pd.Series(data.columns[2:9])
    
    return data, features, targets

data, features, targets = select_features()
data.to_csv("data.csv")
features.to_csv("features.csv")
targets.to_csv("targets.csv")