import benchmark_tests.visualizations.mpso_visuals as vmpso
import benchmark_tests.visualizations.pso_visuals as vpso
import benchmark_tests.visualizations.iteration_visuals as iv
import benchmark_tests.visualizations.benchmark_visuals as bv
import benchmark_tests.visualizations.complete_run_visuals as cv
import multiprocessing
import os
import csv
import time

def make_iteration_visuals():
    path = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-21_13-35-54/tests/mpsoccd-quality-rastrigin/MPSORuns/MPSO-Iteration-0"
    iv.make_iteration_visuals(path)

def bmark_visuals():
    path = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-07 11-04-52/tests"
    for index, i in enumerate(os.listdir(path)):
        new_path = os.path.join(path, i)
        bv.make_benchmark_visuals(new_path)
        print(f"Made graphs for {i}")
        print(f"{index + 1} / {len(os.listdir(path))}")

def mpso_vs_mpsoccd():
    path = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-21_13-35-54"
    function = "shiftedschwefel"
    cv.make_mpso_vs_mpsoccd_graph(path, function)


if __name__ == "__main__":
    make_iteration_visuals()
