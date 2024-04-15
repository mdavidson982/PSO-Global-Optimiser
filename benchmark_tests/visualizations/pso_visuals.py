import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from .utils import _make_printer

def make_pso_visualization(
    path: str, 
    make_time: bool = True, 
    make_quality: bool = True, 
    make_ccd: bool = True,
    verbose: int = 0
):
    """Makes a visualizations for an individual pso runs."""
    printer = _make_printer(verbose)
    dir_files = os.listdir(path)
    if "PSOData.csv" not in dir_files:
        raise Exception("No such csv in folder")
    
    # Create a folder to store the figures if it does not exist.  Lives in the same place as the provided path.
    figures_path = os.path.join(path, "PSOfigures")
    if "PSOfigures" not in dir_files:
        os.mkdir(figures_path)
    
    df: pd.DataFrame = pd.read_csv(os.path.join(path, "PSOData.csv"), index_col=0)

    # We want to separate the mpso iterations out.
    mpso_iterations = df["mpso_iteration"].unique()

    # Determine if there are any CCD rows in the dataframe
    if len(df[df["is_ccd"]==True]) > 0:
        has_ccd = True
    else:
        has_ccd = False
    
    printer("Making visualizations")
    for i, mpso_iteration in enumerate(mpso_iterations):
        printer(f"Making visualization {i + 1} / {mpso_iterations.shape[0]}...", end = "")
        # Grab only data that matches the specific MPSO iteration
        iteration_data = df[df["mpso_iteration"] == mpso_iteration]

        # If there are CCD values, rearrange the data so that only pso information and pso including CCD are stored separately.
        if has_ccd:
            iteration_data_ccd = iteration_data
            iteration_data = iteration_data[iteration_data["is_ccd"] == False]

        # Make the up to four figures.
        if make_quality:
            make_quality_figure_pso(iteration_data, f"PSO-QualityVsIteration-{mpso_iteration}", figures_path, show_ccd = False)
            if has_ccd and make_ccd:
                make_quality_figure_pso(iteration_data_ccd, f"PSO-CCD-QualityVsIteration-{mpso_iteration}", figures_path, show_ccd = True)
        if make_time:
            make_time_figure_pso(iteration_data, f"PSO-TimeVsIteration-{mpso_iteration}", figures_path, show_ccd = False)
            if has_ccd and make_ccd:
                make_time_figure_pso(iteration_data, f"PSO-CCD-TimeVsIteration-{mpso_iteration}", figures_path, show_ccd = True)
        printer("Done!")
        plt.close("all")

def make_quality_figure_pso(df: pd.DataFrame, title: str, path: str, show_ccd: bool):
    """Makes a figure of pso_iteration vs quality"""
    x = np.array(df["pso_iteration"])
    y = np.array(df["g_best_value"])

    fig, ax = _make_figure_pso(x, y, show_ccd)
    ax.set_ylabel("Quality")
    ax.set_title(title)
    plt.savefig(os.path.join(path, title), dpi=600)
    plt.close(fig)
    
def make_time_figure_pso(df: pd.DataFrame, title: str, path: str, show_ccd: bool):
    """Makes a figure of pso_iteration vs time"""
    x = np.array(df["pso_iteration"])
    y = np.array(df["time"])

    fig, ax = _make_figure_pso(x, y, show_ccd)
    ax.set_ylabel("Time")
    ax.set_title(title)
    plt.savefig(os.path.join(path, title), dpi=600)
    plt.close(fig)

def _make_figure_pso(x: np.ndarray, y: np.ndarray, show_ccd: bool):
    """Creates a standard line graph"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # If the below is set to false, all values are treated like they are a regular PSO row.
    # If false, the last row is a CCD row, which should be displayed differently than the others.
    if not show_ccd:
        ax.plot(x, y, linestyle="-", color="b", label="PSO")
    else:
        ax.plot(x[:-1], y[:-1], linestyle="-", color="b", label="PSO")
        ax.plot(x[-2:], y[-2:], linestyle="-", color="r", label="CCD")
    ax.legend()
    ax.set_xlabel("Iteration")
    return fig, ax