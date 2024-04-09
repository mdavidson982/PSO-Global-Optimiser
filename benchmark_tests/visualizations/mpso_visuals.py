import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .utils import _make_printer

def _reformat_df_only(df: pd.DataFrame):
    """Reformat the dataframe for use in mpso_only visualizations"""
    ccd_cols = ["time_ccd", "g_best_coords_ccd", "g_best_value_ccd"]
    df["is_ccd"] = [False] * df.shape[0]

    # If there are any ccd values, reformat these to be intermittent steps between mpso
    if any([ccd_col in df.columns for ccd_col in ccd_cols]):
        new_rows = []
        for row in df:
            row = df.loc[row]
            new_row_regular = {col: row[col] for col in df.columns}
            new_row = {"mpso_iteration": row["mpso_iteration"] + 0.5, "is_ccd": True}
            for ccd_col in ccd_cols:
                if ccd_col in df.columns:
                    new_row[ccd_col[:-len("_ccd")]] = row[ccd_col]
            new_rows.extend([new_row, new_row_regular])
            df = pd.DataFrame(df)
    return df

      
def make_mpso_only_visualization(
    path: str,
    make_time: bool = True,
    make_quality: bool = True,
    verbose: bool = False
):
    """Make MPSO Iteration graphs only showing the result of MPSO and not intermittent PSO steps"""
    printer = _make_printer(verbose)
    dir_files = os.listdir(path)
    if "MPSOData.csv" not in dir_files:
        raise Exception("No MPSOData.csv csv in folder")
    
    figures_path = os.path.join(path, "MPSOfigures")
    if "MPSOfigures" not in dir_files:
        os.mkdir(figures_path)

    df: pd.DataFrame = pd.read_csv(os.path.join(path, "MPSOData.csv"), index_col = 0)
    df = _reformat_df_only(df)

    ccd_values = df[df["is_ccd"] == True]
    if ccd_values.shape[0] > 0:
        has_ccd = True
        ccd_title = " (Before and After CCD)"
    else:
        has_ccd = False
        ccd_title = ""
    
    if make_quality and "g_best_value" in df.columns:
        make_quality_figure_only_mpso(df, title = f"MPSO{ccd_title}-Only-Quality", path = figures_path, before_ccd = True)
        if has_ccd:
            make_quality_figure_only_mpso(df, title = f"MPSO (After CCD)-Only-Quality", path = figures_path, before_ccd = False)
    if make_time and "time" in df.columns:
        make_time_figure_only_mpso(df, title = f"MPSO{ccd_title}-Only-Time", path = figures_path, before_ccd = True)
        if has_ccd:
            make_time_figure_only_mpso(df, title = f"MPSO (After CCD)-Only-Time", path = figures_path, before_ccd = True)
    plt.close("all")
    
def make_quality_figure_only_mpso (df: pd.DataFrame, title: str, path: str, before_ccd: bool = False):
    fig, ax = _make_figure_only_mpso(df, "g_best_value", before_ccd = before_ccd)
    ax.set_ylabel("Quality")
    ax.set_title(title)

    plt.savefig(os.path.join(path, title), dpi=600)
    plt.close(fig)

def make_time_figure_only_mpso (df: pd.DataFrame, title: str, path: str, before_ccd: bool = False):
    fig, ax = _make_figure_only_mpso(df, "time", before_ccd = before_ccd)
    ax.set_ylabel("Time")
    ax.set_title(title)

    plt.savefig(os.path.join(path, title), dpi=600)
    plt.close(fig)

def _make_figure_only_mpso(df: pd.DataFrame, ylabel, before_ccd: bool):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if before_ccd:
        ax.plot(np.array(df["mpso_iteration"]), np.array(df[ylabel]), linestyle = "-", color="g", label="MPSO iterations")
        ccd_vals = 

        ax.scatter(np.array(df["mpso_i"]))

    ax.plot(np.array(df["mpso_iteration"]), np.array(df[ylabel]), linestyle = "-", color="g", label="MPSO iterations")

    if before_ccd

    if plot_ccd and show_ccd:
        ax.scatter(np.array(ccd_values.index), np.array(ccd_values[ylabel]), marker = "x", color = "red", label = "CCD Points")
    
    uniques = df["mpso_iteration"].unique()
    #ax.scatter(np.array(mpso_labels.index), np.array(mpso_labels[ylabel]), marker = "o", color = "orange", label = "New MPSO Iteration")

    ax.legend()
    ax.set_xlabel("Iteration")
    return fig, ax

    

def _reformat_df_full(df: pd.DataFrame):
    if "time" in df.columns:
        for val in sorted(df["mpso_iteration"].unique(), reverse=True):
            add_time = df[df["mpso_iteration"] == val]["time"].max()
            df.loc[df["mpso_iteration"]> val, "time"] += add_time

def make_full_mpso_visualizations(
    path: str,
    make_time: bool = True,
    make_quality: bool = True,
    verbose: bool = False
):
    """Make an mpso iteration graph with all intermediate pso iterations visualized"""
    printer = _make_printer(verbose)
    dir_files = os.listdir(path)
    if "PSOData.csv" not in dir_files:
        raise Exception("No PSOData.csv csv in folder")
    figures_path = os.path.join(path, "MPSOfigures")
    if "MPSOfigures" not in dir_files:
        os.mkdir(figures_path)

    df: pd.DataFrame = pd.read_csv(os.path.join(path, "PSOData.csv"), index_col=0)
    _reformat_df_full(df)

    if make_quality:
        printer("Making quality...", end = "")
        make_quality_figure_full_mpso(df, "MPSO-Full-Quality", figures_path)
        printer("Done!")
    if make_time:
        printer("Making time...", end = "")
        make_time_figure_full_mpso(df, "MPSO-Full-Time", figures_path)
        printer("Done!")
    plt.close("all")
    

def make_quality_figure_full_mpso(df: pd.DataFrame, title: str, path: str):
    fig, ax = _make_figure_full_mpso(df, "g_best_value", show_ccd = True, show_mpso_iterations = True)
    ax.set_ylabel("Quality")
    ax.set_title(title)

    plt.savefig(os.path.join(path, title), dpi=600)
    plt.close(fig)
    
def make_time_figure_full_mpso(df: pd.DataFrame, title: str, path: str):
    fig, ax = _make_figure_full_mpso(df, "time", show_ccd = True, show_mpso_iterations = True)
    ax.set_ylabel("Time")
    ax.set_title(title)

    plt.savefig(os.path.join(path, title), dpi=600)
    plt.close(fig)

def _make_figure_full_mpso(df: pd.DataFrame, ylabel, show_ccd: bool = True, show_mpso_iterations: bool = True):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ccd_values = df[df["is_ccd"] == True]
    if ccd_values.shape[0] > 0:
        plot_ccd = True
    else:
        plot_ccd = False
    ax.plot(np.array(df.index), np.array(df[ylabel]), linestyle = "-", color="g", label="MPSO showing PSO intermediate values")
    if plot_ccd and show_ccd:
        ax.scatter(np.array(ccd_values.index), np.array(ccd_values[ylabel]), marker = "x", color = "red", label = "CCD Points")
    
    uniques = df["mpso_iteration"].unique()
    
    mpso_labels = df.loc[[df.index[df["mpso_iteration"] == unique].min() for unique in uniques]]
    if show_mpso_iterations:
        for index, i in enumerate(mpso_labels.index):
            label = mpso_labels.loc[i]["mpso_iteration"]
            y = mpso_labels.loc[i][ylabel]

            annot = ax.annotate(f"{label}",
                xy = (i, y),
                xytext = (0, 200),
                textcoords = "offset pixels",
                rotation = -90,
                fontsize = 8,
                ha = "center",
                arrowprops=dict(arrowstyle = "-", color='orange'),
            )

            if index == 0:
                annot.set_text(f"MPSO\nIt.{label}")

    #ax.scatter(np.array(mpso_labels.index), np.array(mpso_labels[ylabel]), marker = "o", color = "orange", label = "New MPSO Iteration")

    ax.legend()
    ax.set_xlabel("Iteration")
    return fig, ax
