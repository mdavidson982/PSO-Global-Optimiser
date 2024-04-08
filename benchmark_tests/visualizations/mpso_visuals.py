import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .utils import _make_printer

def make_full_mpso_visualization(
    path: str,
    make_time: bool = True,
    make_quality: bool = True,
    use_pso_iterations: bool = True,
    verbose: bool = False
):
    """Make an mpso iteration with all intermediate pso iterations visualized"""
    printer = _make_printer(verbose)
    dir_files = os.listdir(path)
    if "PSOData.csv" not in dir_files:
        raise Exception("No such csv in folder")
    figures_path = os.path.join(path, "MPSOfigures")
    if "MPSOfigures" not in dir_files:
        os.mkdir(figures_path)

    df: pd.DataFrame = pd.read_csv(os.path.join(path, "PSOData.csv"), index_col=0)
    if make_quality:
        make_quality_figure_mpso(df, "MPSO-Quality", figures_path)
    if make_time:
        pass
    
        



def make_quality_figure_mpso(df: pd.DataFrame, title: str, path: str):
    fig, ax = plt.subplots()








    
    ax.set_ylabel("Quality")

    plt.savefig(os.path.join(path, title), dpi=600)
    plt.close(fig)
    
def make_time_figure_mpso(df: pd.DataFrame, title: str, path: str, show_ccd: bool):
    x = np.array(df["pso_iteration"])
    y = np.array(df["time"])

    fig, ax = _make_figure_pso(x, y, show_ccd)
    ax.set_ylabel("Time")
    plt.savefig(os.path.join(path, title), dpi=600)
    plt.close(fig)

def _make_figure_pso(df: pd.DataFrame, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ccd_values = df[df["is_ccd"] == True]
    if ccd_values.shape[0] > 0:
        plot_ccd = True
    ax.plot(df["pso_iteration"], df[ylabel], linestyle = "-", color="g", label="MPSO")
    if plot_ccd:
        ax.scatter(ccd_values["pso_iteration"], ccd_values[ylabel], marker = "x", color = "red", label = "CCD Points")

    
    ax.legend()
    ax.set_xlabel("Iteration")
    return fig, ax




z = {
 "A": [4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 0],
}

df = pd.DataFrame(z)
print(df)

uniques = df["A"].unique()
for unique in uniques:
    print(df.index[df["A"] == unique].min())
