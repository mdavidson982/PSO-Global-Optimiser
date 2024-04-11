from . import mpso_visuals as vmpso
from . import pso_visuals as vpso
import os
from .utils import _make_printer

def make_iteration_visuals(
    path: str,
    make_time: bool = True,
    make_quality: bool = True,
    verbose: int = 0
):
    printer = _make_printer(verbose)
    dir_files = os.listdir(path)
    print(path)
    input()

    if "PSOData.csv" not in dir_files and "MPSOData.csv" not in dir_files:
        raise Exception("No visuals to generate!")

    vmpso.make_all_mpso_visualizations_for_iteration(path, make_time, make_quality, verbose - 1)
    
    print(str("PSOData.csv" in dir_files) + " Yup")
    if "PSOData.csv" in dir_files:
        print("here")
        #vpso.make_pso_visualization(path, make_time, make_quality, make_ccd = True, verbose = verbose - 2)
    
def make_all_iterations_visuals(
    path: str,
    make_time: bool = True,
    make_quality: bool = True,
    verbose: int = 0
):
    printer = _make_printer(verbose)
    dir_files = os.listdir(path)

    # Allows the user to refer to the test name (e.g. mpso-quality-rosenbrock) or the actual MPSORuns folder
    if "MPSORuns" in dir_files and os.path.isdir(os.path.join(path, "MPSORuns")):
        path = os.path.join(path, "MPSORuns")
        dir_files = os.listdir(path)


    for file in dir_files:
        printer(f"Making visuals for {file}", end = "...")
        make_iteration_visuals(os.path.join(path, file), make_time, make_quality, verbose)
        printer("Done!")