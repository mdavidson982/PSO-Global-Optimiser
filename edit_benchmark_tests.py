import benchmark_tests.utils as u
import pso.psodataclass as dc


def run_edit_configs_all_functions():
    z = {"ccd_max_its":10}
    u.edit_configs_all_functions(dc.CCDHyperparameters, z)

run_edit_configs_all_functions()