# Figure 1

# Read data
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", index_col = 0)

# Aggregate per hour, per day, across individuals to reproduce Aledavood et al. (2020) Fig 1
data["time"] = pd.to_datetime(data["time"])
data.set_index("time", inplace=True)
data = data.groupby("id").resample("H").durationSeconds.sum()
data = pd.DataFrame(data.reset_index())
data["weekday"] = pd.to_datetime(data.time).dt.dayofweek
data["HH"] = pd.to_datetime(data.time).dt.hour
data["date"] = pd.to_datetime(data["time"]).dt.date

# Drop dates without any smartphone usage
data = pd.merge(data, data.groupby(["id","date"]).sum().reset_index()[["id","date","durationSeconds"]], on = ["id","date"], how = "outer")
data = data[data.durationSeconds_y != 0]

# Select relevant variables and rename durationSeconds_x
data = data[["id","weekday","HH","durationSeconds_x"]]
data.rename({"durationSeconds_x":"durationSeconds"}, axis=1, inplace=True)

# Calculate average hourly time spent on smartphone per weekday 
hourly_use = pd.DataFrame(data.groupby(["weekday","HH"]).durationSeconds.mean())

# Normalize smartphone usage duration
normalized_use = hourly_use.durationSeconds.values/sum(hourly_use.durationSeconds.values)

# Visualize normalized smartphone usage duration across the week
import seaborn as sns
sns.set_theme()
fig = plt.figure(figsize=(20,5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(hourly_use.values)
ax.set_xticks(range(0,164,24))
ax.set_xticklabels(['Mo','Tu','We','Th','Fr','Sa','Su'])
plt.xlim(0,170)
plt.ylim(0,900)
plt.xlabel("Hours since start of week")
plt.ylabel("Mean time on smartphone applications (seconds)")
plt.vlines(range(0,168,24),0,900,color="red")
plt.vlines(range(12,168,24),0,900,color="gray",linestyles="dashed")
plt.title("Mean time on smartphone applications per hour (aggregated across participants)")
plt.savefig("/home/haalbers/sleep-estimation/gen/paper/figure-smartphone-use-across-the-week.png")
plt.clf()