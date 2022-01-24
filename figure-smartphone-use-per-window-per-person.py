# Import modules
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

# Read data
print("Reading the data.")
log_data = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", index_col = 0, nrows = 200000)
log_data.index = pd.to_datetime(log_data["time"])
log_data["durationSeconds"] = (pd.to_datetime(log_data["dt_e"]) - pd.to_datetime(log_data["dt_s"])).dt.seconds

# One data point has negative duration, so we remove it
log_data = log_data[log_data["durationSeconds"] <= 900]
data = log_data.groupby("id").resample("15Min").durationSeconds.sum().reset_index()
data["Date"] = pd.to_datetime(data["time"]).dt.date
data["minute"] = pd.to_datetime(data["time"]).dt.minute
data["hour"] = pd.to_datetime(data["time"]).dt.hour
data["Time"] = data["hour"] + data["minute"]/60
data = data.pivot(values = "durationSeconds", index = ["id","Date"], columns = "Time").fillna(0).reset_index()

# Create plot for each participant
sns.set(rc={"figure.figsize":(10, 10)}, font_scale=4)
palette = sns.color_palette("mako", as_cmap=True)
for pp in ["User #24612", "User #24225", "User #24198", "User #15258"]:
    print("Creating panel for participant", pp)
    pp_data = data[(data.reset_index().id == pp).values]
    pp_data.set_index(["Date"], inplace=True)
    ax = sns.heatmap(pp_data.iloc[:,1:], cmap = palette, cbar = False, xticklabels = False, yticklabels = False)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig("/home/haalbers/sleep-estimation/gen/paper/smartphone-use-participant-" + str(pp) + ".png", bbox_inches='tight')
    plt.clf()