import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def make_pso_visualization(path: str, make_time: bool, make_quality: bool):
    dir_files = os.listdir(path)
    if "PSOData.csv" not in dir_files:
        raise Exception("No such csv in folder")
    
    figures_path = os.path.join(path, "PSOfigures")
    if "PSOfigures" not in dir_files:
        os.mkdir(figures_path)

    df: pd.DataFrame = pd.read_csv(os.path.join(path, "PSOData.csv"), index_col=0)
    mpso_iterations = df["MPSOIteration"].unique()
    for mpso_iteration in mpso_iterations:
        iteration_data = df[df["MPSOIteration"] == mpso_iteration]
        if make_quality:
            make_quality_figure_pso(iteration_data, f"PSO-QualityVsIteration-{mpso_iteration}", figures_path)
        if make_time:
            make_time_figure_pso(iteration_data, f"PSO-TimeVsIteration-{mpso_iteration}", figures_path)

        
def make_quality_figure_pso(df, title, path):
    df = df[df["is_ccd"]==False]
    x = np.array(df["pso_iteration"])
    y = np.array(df["g_best_value"])
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Quality")
    plt.savefig(os.path.join(path, title), dpi=600)

def make_quality_figure_psoccd(df, title, path):
    x = np.array(df["pso_iteration"])
    y = np.array(df["g_best_value"])
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Quality")
    plt.savefig(os.path.join(path, title), dpi=600)
    
def make_time_figure_pso(df, title, path):
    x = np.array(df["pso_iteration"])
    y = np.array(df["time"])
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Quality")
    plt.savefig(os.path.join(path, title), dpi=600)