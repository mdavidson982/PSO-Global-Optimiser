import benchmark_tests.visualizations.mpso_visuals as vmpso
import benchmark_tests.visualizations.pso_visuals as vpso
import os

z = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-08 14-20-35/tests/mpsoccd-quality-sphere/MPSORuns/MPSO-Iteration-0"
#v.make_pso_visualization(path = z,verbose = True)
#vmpso.make_full_mpso_visualization(path = z)


z = "/home/jcm/Documents/PSO/PSO-Global-Optimiser/benchmark_tests/benchmarkruns/Benchmark2024-04-09 09-23-07/tests"
for i in [directory for directory in os.listdir(z) if "time" not in directory]:
    print(f"Making visuals for {i}")
    new_path = os.path.join(z, i, "MPSORuns", "MPSO-Iteration-0")
    vmpso.make_full_mpso_visualizations(new_path, verbose = True)
    vpso.make_pso_visualization(new_path, verbose = True)
