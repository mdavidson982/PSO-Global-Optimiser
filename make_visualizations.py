import benchmark_tests.visualizations.mpso_visuals as vmpso
import benchmark_tests.visualizations.pso_visuals as vpso
import benchmark_tests.visualizations.iteration_visuals as iv
import multiprocessing
import os
import csv


z = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-11_10-25-11/tests"
workers = len(os.listdir(z))
for i in os.listdir(z):
    new_path = os.path.join(z, i)
    process = multiprocessing.Process(target = iv.make_all_iterations_visuals, kwargs={"path": new_path, "verbose": 3})
    process.start()
    #iv.make_all_iterations_visuals(new_path, verbose = 3)
