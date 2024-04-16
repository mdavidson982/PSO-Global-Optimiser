import os
import matplotlib.pyplot as plt
import pandas as pd
from pso.codec import json_file_to_dataclass
from pso.psodataclass import FunctionData
from .utils import _make_printer
from benchmark_tests.utils import dataclasses


def make_benchmark_visuals(path: str, verbose: int = 0):
    printer = _make_printer(verbose)

    dir_files = os.listdir(path)
    if "MPSORuns.csv" not in dir_files:
        raise Exception(f"{path} does not contain MPSORuns.csv")
    
    _, title = os.path.split(path)
    df = pd.read_csv(os.path.join(path, "MPSORuns.csv"))
    try:
        with open(dataclasses[FunctionData]+".json") as file:
            fndata: FunctionData = json_file_to_dataclass(file)
    except Exception:
        with open(os.path.join(path, "domain_data.json")) as file:
            fndata: FunctionData = json_file_to_dataclass(file)
    true_bias = fndata.bias
    true_optimum = fndata.optimum

    skip_quality = False
    if "g_best_value_ccd" in df.columns:
        quality_title = title + " Quality with CCD"
        x = df["g_best_value_ccd"]

    elif "g_best_value" in df.columns:
        quality_title = title + " Quality"
        x = df["g_best_value"]
    else:
        printer("Skipping visuals graphs")
        skip_quality = True
    if not skip_quality:
        make_visuals(x, title = quality_title, savepath = path)
    
    skip_time = False
    if "time_ccd" in df.columns:
        time_title = title + " Time with CCD"
        x = df["time_ccd"]
    elif "time" in df.columns:
        time_title = title + " Time"
        x = df["time"]
    else:
        printer("Skipping time graphs")
        skip_time = True
    if not skip_time:
        make_visuals(x, title = time_title, savepath = path)

def make_visuals(series: pd.Series, title: str, savepath: str):
    make_histogram(series, title, savepath)
    make_boxplot(series, title, savepath)

def make_histogram(x: pd.Series, title: str, savepath: str):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(x, color = "cyan", edgecolor = "black")
    ax.set_title(title)
    ax.set_xlabel(x.name)
    ax.set_ylabel("Number of replicates")
    save = os.path.join(savepath, title + "-histogram.png")
    fig.savefig(save, dpi = 600)
    plt.close("all")
    

def make_boxplot(x: pd.Series, title: str, savepath: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = ax.boxplot(x, patch_artist=True)
    box["boxes"][0].set(color = "blue", linewidth = 1.2)
    box["boxes"][0].set(facecolor = "cyan")

    ax.set_title(title)
    ax.set_ylabel(x.name)
    save = os.path.join(savepath, title + "-boxplot.png")
    
    fig.savefig(save, dpi = 600)
    plt.close("all")