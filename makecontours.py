import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from testfuncts import testfuncts as tf
from benchmark_tests.benchmark_tests import IGNORELIST
import os
from mpso_ccd import codec, psodataclass as dc

save_path = os.path.join("/home/jcm/Downloads/ContourPlots2")
config_path = os.path.join("benchmark_tests", "configs")
colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'jet', 'rainbow']

redo = True
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    print(save_path)

def make_figs(fnname):

    print(f"Making figures for {fnname}")
    func_config = os.path.join(config_path, fnname, "domain_data.json")
    with open(func_config) as file:
        fndata: dc.FunctionData = codec.json_file_to_dataclass(file)
    fndata.lower_bound = fndata.lower_bound[:2]
    fndata.upper_bound = fndata.upper_bound[:2]
    fndata.optimum = fndata.optimum[:2]
    fn = tf.TestFuncts.generate_function(fnname, domaindata = fndata)
    X, Y, Z = tf.TestFuncts.generate_contour(fn, fndata.lower_bound, fndata.upper_bound)

    for colormap in colormaps:
        regular_name = f"{fnname}-{colormap}.png"
        if os.path.exists(regular_name) and not redo:
            print(f"Not redoing {regular_name}")
            continue

        fig = plt.figure()
        contourf = plt.contourf(X, Y, Z, cmap = colormap)
        plt.contour(X, Y, Z, colors = "black")

        fig.set_size_inches(6, 6)
        plt.savefig(os.path.join(save_path, f"{fnname}-{colormap}-clean.png"))

        plt.colorbar(contourf)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{fnname} contour")
        
        fig.set_size_inches(8, 6)
        plt.savefig(os.path.join(save_path, regular_name))

        plt.close("all")
    print(f"Done making figures for {fnname}")




for fnname in [fn for fn in tf.TESTFUNCTSTRINGS if fn not in IGNORELIST]:
    make_figs(fnname)