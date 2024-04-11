import benchmark_tests.visualizations.mpso_visuals as vmpso
import benchmark_tests.visualizations.pso_visuals as vpso
import benchmark_tests.visualizations.iteration_visuals as iv
import os


z = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-11_10-25-11/tests"
for i in os.listdir(z):
    new_path = os.path.join(z, i)
    iv.make_all_iterations_visuals(new_path, verbose = 3)