#import benchmark_tests.visualizations.mpso_visuals as vmpso
import benchmark_tests.visualizations.table_visuals as tv
#import benchmark_tests.visualizations.pso_visuals as vpso
import matplotlib.pyplot as plt
import os
import csv

#search_file = "C:\\Users\\80sbl\\Documents\\AA Shit to look back at\\SeinorSeminarProject\\PSO-Global-Optimiser-1\\benchmark_tests\\benchmarkruns\\Benchmark2024-04-09_14-53-12\\Results.csv"
search_file = os.path.join("c:\\", "Users", "80sbl", "Documents", "AA Shit to look back at", "SeinorSeminarProject", "PSO-Global-Optimiser-1", "benchmark_tests", "benchmarkruns", "Benchmark2024-04-09_14-53-12", "Results.csv")
print(os.getcwd())
print(search_file)
tv.create_visual_table(search_file)

#z = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-08 14-20-35/tests/mpsoccd-quality-sphere/MPSORuns/MPSO-Iteration-0"
#v.make_pso_visualization(path = z,verbose = True)
#vmpso.make_full_mpso_visualization(path = z)

"""
z = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-09 09-23-07/tests"
for i in [directory for directory in os.listdir(z) if "time" not in directory]:
    print(f"Making visuals for {i}")
    new_path = os.path.join(z, i, "MPSORuns", "MPSO-Iteration-0")
    vmpso.make_full_mpso_visualizations(new_path, verbose = True)
    vpso.make_pso_visualization(new_path, verbose = True)
    """