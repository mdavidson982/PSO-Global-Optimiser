import os
import visualization.visualization as v
import pso.codec as codec
import pso.pso as pSO
import tkinter as tk
import testfuncts.testfuncts as tf
import numpy as np

def TestVisualizer(input: str):
    path = os.path.join("./visualization", "configs", input)
    root = tk.Tk()
    
    with open(os.path.join(path, "pso_hyperparameters.json")) as file:
        pso_hyperparameters = codec.json_file_to_dataclass(file)

    with open(os.path.join(path, "domain_data.json")) as file:
        domain_data = codec.json_file_to_dataclass(file)

    with open(os.path.join(path, "pso_config.json")) as file:
        pso_configs = codec.json_file_to_dataclass(file)

    fn = tf.TestFuncts.generate_function(input, domaindata = domain_data)
    print(domain_data)

    pso = pSO.PSO(
        pso_hyperparameters=pso_hyperparameters,
        domain_data=domain_data,
        pso_configs=pso_configs,
        function=fn
    )

    vis = v.Visualization(root=root, pso=pso, update_time = 1000)
    
    vis.start()
    root.mainloop()

if __name__ == "__main__":
    name = "rastrigin"
    TestVisualizer(name)