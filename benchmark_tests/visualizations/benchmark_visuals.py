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
    df = pd.read_csv("MPSORuns.csv")
    try:
        with open(dataclasses[FunctionData]+".json") as file:
            fndata = FunctionData(json_file_to_dataclass(file))
    except Exception:
        with open(dataclasses["domain_data.json"]) as file:
            fndata = FunctionData(json_file_to_dataclass(file))
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
        skip_quality = True
    make_visuals(x, title = quality_title)
    
    if "time_ccd" in df.columns:
        time_title = title + " Time with CCD"
        x = df["time_ccd"]
        make_visuals()
    elif "time" in df.columns:
        x = df
        make_visuals()


def make_visuals(series: pd.Series, qual_title: str, time_title: str):
    make_histogram()

def make_histogram(x: pd.Series, title):
    fig = plt.figure()
    plt.hist(x, bins = len(x), color = "blue")
    
    

    pass

def make_boxplot():
    pass