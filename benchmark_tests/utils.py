import pso.psodataclass as dc
from .benchmark_tests import CONFIG_FOLDER
import os
import pso.codec

dataclasses = {
    dc.CCDHyperparameters: "ccd_hyperparameters",
    dc.FunctionData: "function_data",
    dc.MPSOConfigs: "mpso_config",
    dc.MPSOLoggerConfig: "mpso_logger_config",
    dc.PSOConfig: "pso_config",
    dc.PSOHyperparameters: "pso_hyperparameters",
    dc.PSOLoggerConfig: "pso_logger_config",
}

def edit_specific_function(func_name: str, type: type, new: dict, file_path:str = None):
    """ Edit a specific function's config in behcmark_tests/configs."""
    if type not in list(dc.DATACLASSES.values()):
        raise Exception("Not a valid type to change")
    
    if file_path == None:
        file_path = os.path.join(CONFIG_FOLDER, func_name, dataclasses[type] + ".json")

    with open(file_path, "r") as file:
        dclass = pso.codec.json_file_to_dataclass(file)
        for key in new:
            if not hasattr(dclass, key):
                raise Exception(f"{key} not a valid field")
            setattr(dclass, key, new[key])
    with open(file_path, "w+") as file:
        pso.codec.dataclass_to_json_file(dclass, file)

def edit_configs_all_functions(type: type, new: dict, folder_path: str = None):
    """
    Edit the configs of a specific datatype for all classes.

    E.g. if type = pso_hyperparameters, and new = {num_part: 90}, this would set the number of particles in all configs to 90.
    
    """
    if type not in list(dc.DATACLASSES.values()):
        raise Exception("Not a valid type to change")
    
    if folder_path == None:
        folders = os.listdir(CONFIG_FOLDER)
    else:
        folders = os.listdir(folder_path)
    for folder in folders:
        if folder_path == None:
            path = None
        else:
            path = os.path.join(folder_path, folder, dataclasses[type] + ".json")
        edit_specific_function(folder, type, new, path)
