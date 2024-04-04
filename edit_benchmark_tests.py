import benchmark_tests.utils as u
import pso.psodataclass as dc
import numpy as np


def run_edit_configs_all_functions():
    #z = {"num_part":100}
    #u.edit_configs_all_functions(dc.PSOHyperparameters, z)

    #z = {"ccd_max_its": 10, "third_term_its": 3}
    #u.edit_configs_all_functions(dc.CCDHyperparameters, z)

    z = {"iterations": 15}
    u.edit_configs_all_functions(dc.MPSOConfigs, z)

def run_edit_specific_function():
    upper = np.ones(30)*5
    lower = -upper

    z = {"upper_bound": upper, "lower_bound": lower, "bias":-330}
    
    function = "rotatedrastrigin"

    u.edit_specific_function(function, dc.FunctionData, z)
    

#run_edit_configs_all_functions()
run_edit_specific_function()