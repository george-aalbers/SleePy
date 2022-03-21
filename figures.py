# Import modules
import pandas as pd
import numpy as np
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import os
import pickle
from evaluate_models import return_multiple_rho, return_multiple_mae

def figure_1():

    # Figure 1

    # Read data
    data = pd.read_csv("/home/haalbers/dissertation/mobiledna-clean.csv", index_col = 0)
    study_parameters = pd.read_json("study_parameters.json")

    # Aggregate per hour, per day, across individuals to reproduce Aledavood et al. (2020) Fig 1
    data["time"] = pd.to_datetime(data["time"])
    data.set_index("time", inplace=True)
    data = data.groupby("id").resample("15Min").durationSeconds.sum()
    data = pd.DataFrame(data.reset_index())
    data["weekday"] = pd.to_datetime(data.time).dt.dayofweek
    data["hour"] = pd.to_datetime(data.time).dt.hour
    data["minute"] = pd.to_datetime(data.time).dt.minute
    data["date"] = pd.to_datetime(data["time"]).dt.date

    # Drop dates without any smartphone usage
    data = pd.merge(data, data.groupby(["id","date"]).sum().reset_index()[["id","date","durationSeconds"]], on = ["id","date"], how = "outer")
    data = data[data.durationSeconds_y != 0]

    # Select relevant variables and rename durationSeconds_x
    data = data[["id","weekday", "hour", "minute","durationSeconds_x"]]
    data.rename({"durationSeconds_x":"durationSeconds"}, axis=1, inplace=True)

    # Calculate average quarterly time spent on smartphone per weekday 
    quarterly_use = pd.DataFrame(data.groupby(["weekday","hour","minute"]).durationSeconds.mean())

    # Normalize smartphone usage duration
    normalized_use = hourly_use.durationSeconds.values/sum(hourly_use.durationSeconds.values)

    # Visualize normalized smartphone usage duration across the week
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(quarterly_use.values, color = "#36a0ac")
    ax.set_xticks(range(0,672,96))
    ax.set_xticklabels(['Mo','Tu','We','Th','Fr','Sa','Su'], fontdict={'fontsize': 24})
    ax.set_yticks(range(0,300,100))
    ax.set_yticklabels(['0', '100', '200'], fontdict={'fontsize': 24})
    plt.xlim(-10,672)
    plt.ylim(0,200)
    plt.xlabel("Time (15-minute windows)", fontdict={'fontsize': 24})
    plt.ylabel("Average smartphone app usage (seconds)", fontdict={'fontsize': 24})
    plt.vlines(range(96,672,96),0,900,color="black")
    plt.vlines(range(48,672,96),0,900,color="black",linestyles="dashed")
    plt.title("Average time on smartphone applications per 15-minute window", fontdict={'fontsize': 24})
    plt.tight_layout()
    plt.savefig(study_parameters["markdown_path"][0] + "figure_1.png")
    plt.clf()

def figure_2():    
    study_parameters = pd.read_json("study_parameters.json")
    
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
    for pp in ["User #24198", "User #15258"]:
        print("Creating panel for participant", pp)
        pp_data = data[(data.reset_index().id == pp).values]
        pp_data.set_index(["Date"], inplace=True)
        pp_data = pp_data.iloc[:130,:]
        ax = sns.heatmap(pp_data.iloc[:,1:], cmap = palette, cbar = False, xticklabels = False, yticklabels = False)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.savefig(study_parameters["markdown_path"][0] + "smartphone-use-participant-" + str(pp) + ".png", bbox_inches='tight')
        plt.clf()
    
def figure_3():

    study_parameters = pd.read_json("study_parameters.json")
    root = os.getcwd()
    data = pd.DataFrame()
    for row, study in study_parameters.iterrows():
        for i in range(1,5,1):
            model_name = os.listdir(study["model_output_path"])[0].split("_")[0]

            filename = study["model_output_path"] + model_name + "_" + str(i) + ".pkl"

            model = pickle.load(open(filename, 'rb'))
            X_test = pd.read_csv(study["data_output_path"] + "/X_test_" + str(i) + ".csv", index_col = 0)
            y_test = pd.read_csv(study["data_output_path"] + "/y_test_" + str(i) + ".csv", index_col = 0)

            y_pred = model.predict(X_test)
            y_pred = pd.DataFrame(y_pred)
            y_pred.columns = ["y_pred"]
            y_test.columns = ["y_test"]

            df = pd.concat([y_test.reset_index(), y_pred], axis = 1)
            data = pd.concat([data, df], axis = 0)

        data.reset_index(inplace=True,drop=True)

        sns.set_theme()
        sns.scatterplot(
            data=data,
            x="y_test",
            y="y_pred", 
            color="k"
        )
        plt.xlabel("Self-reported sleep duration")
        plt.ylabel("Estimated sleep duration")
        plt.xlim(2,12)
        plt.ylim(2,12)
        plt.title("Prediction error plot of " + model_name.upper())
        sns.kdeplot(
            data=data,
            x="y_test",
            y="y_pred",
            levels=5,
            fill=True,
            alpha=0.6,
            cut=2
        )
        plt.axline((0, 0), (1, 1), linewidth=4, color='r')
        plt.savefig(study["markdown_path"] + "prediction_error_plot_" + model_name.upper() + ".png")
        plt.clf()

def figure_4():

    sns.set_theme()
    study_parameters = pd.read_json("study_parameters.json")
    for row, study in study_parameters.iterrows():

        model_name = study["model_type"].upper()
        df = return_multiple_rho(row + 1, study["model_type"])
        df.columns = ["rho"]
        sns.histplot(df.rho.values)
        plt.xlabel("Person-specific Spearman rank-order correlation")
        plt.ylabel("Number of participants")
        plt.xlim(-0.3,1)
        plt.ylim(0,50)
        plt.title("Distribution of Spearman rho for " + model_name)
        plt.savefig(study["markdown_path"] + study["model_type"] + "_histogram_rho.png")
        plt.clf()

def figure_5():
    
    study_parameters = pd.read_json("study_parameters.json")
    for row, study in study_parameters.iterrows():

        model_name = study["model_type"].upper()
        df = return_multiple_mae(row + 1, study["model_type"])
        df.columns = ["mae"]
        sns.histplot(df.mae.values * 60)
        plt.xlabel("Person-specific median absolute error")
        plt.ylabel("Number of participants")
        plt.xlim(0,2.5)
        plt.ylim(0,40)
        plt.title("Distribution of MAE for " + model_name)
        plt.savefig(study["markdown_path"] + study["model_type"] + "_histogram_mae.png")
        plt.clf()
        
def plot_figures():
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()