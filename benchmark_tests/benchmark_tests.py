import time
from mpso_ccd import pso as pso_file, mpso as mpso_file, psodataclass as dc, codec
import testfuncts.testfuncts as tf
import numpy as np
import pandas as pd
import os
import itertools
import json
import utils.util as u

from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

QUALITY = 0
TIME = 1

TRACK_TYPES = [QUALITY, TIME]

MPSO = 0
MPSOCCD = 1

MPSO_TYPES = [MPSO, MPSOCCD]

RESULTSFOLDER = os.path.join("benchmark_tests", "benchmarkruns")
CONFIG_FOLDER = os.path.join("benchmark_tests", "configs")

IGNORELIST = [tf.SHIFTEDELLIPTICSTRING, 
              tf.GRIEWANKSTRING, 
              tf.SHIFTEDROTATEDACKLEYSTRING,
              tf.ROTATEDRASTRIGINSTRING,
]

def _make_benchmark_folder():
    """Makes a run for the overall benchmark folder"""
    formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(RESULTSFOLDER, f"Benchmark{formatted_datetime}")
    os.makedirs(folder_path)
    return folder_path

def _make_benchmark_test_folder(benchmark_path: str, extension: str):
    """Makes a directory for an individual test"""
    extension_path = os.path.join(benchmark_path, extension)
    os.mkdir(extension_path)
    return extension_path

def _make_replicate_folder(benchmark_test_path: str, replicate: int):
    """Makes a directory for an individual replicate"""
    replicate_path = os.path.join(benchmark_test_path, f"MPSO-Iteration-{replicate}")
    os.mkdir(replicate_path)
    return replicate_path


def output_configs(logger: mpso_file.MPSOLogger, extension_path: str):
    """Output all of the configs that a given MPSO used to a folder"""
    configs = [
        (logger.config, "mpso_logger_config"),
        (logger.mpso.mpso_config, "mpso_config"),
        (logger.mpso.ccd_hyperparameters, "ccd_hyperparameters"),
        (logger.mpso.pso.pso.domain_data, "domain_data"),
        (logger.mpso.pso.pso.pso_configs, "pso_config"),
        (logger.mpso.pso.pso.pso_hypers, "pso_hyperparameters"),
    ]

    if type(logger.mpso.pso) == pso_file.PSOLogger:
        configs.append((logger.mpso.pso.config, "pso_logger_config"))

    for config in configs:
        with open(os.path.join(extension_path, config[1]+".json"), "w+") as file:
            codec.dataclass_to_json_file(config[0], file)


def record_mpso_run(logger: mpso_file.MPSOLogger, extension_path: str, iteration: int):
    """Record an individual mpso replicate"""
    replicate_path = _make_replicate_folder(extension_path, iteration)
    pso_path = os.path.join(replicate_path, "PSOData.csv")
    if "PSOData" in list(logger.rows[-1].keys()): # If PSO data was not tracked, it will not appear in the result data
        rows = []
        for i, row in enumerate(logger.rows):
            if "PSOData" not in row.keys():
                continue
            pso_rows = row["PSOData"]
            for j in range(len(pso_rows)):
                pso_rows[j]["mpso_iteration"] = i+1
                pso_rows[j]["is_ccd"] = False
                pso_rows[j]["pso_iteration"]
            rows.extend(pso_rows)

            if logger.config.track_ccd:
                # If tracking CCD, append a special row to show its results.
                ccd_row = {
                    "mpso_iteration": i+1,
                    "is_ccd":  True,
                    "pso_iteration": len(pso_rows)
                }
                if logger.config.track_time:
                    ccd_row.update({
                        "time": row["time_ccd"]
                    })

                if logger.config.track_quality:
                    ccd_row.update({
                        "g_best_value": row["g_best_value_ccd"],
                        "g_best_coords": row["g_best_coords_ccd"]
                    })
                rows.append(ccd_row)
            # Once recorded, delete this to free up memory.
            del logger.rows[i]["PSOData"]
        pd.DataFrame(rows).to_csv(pso_path)
    mpso_path = os.path.join(replicate_path, "MPSOData.csv")
    pd.DataFrame(logger.rows).to_csv(mpso_path)
        

def perform_statistical_analysis(df: pd.DataFrame, track_name, mpso_name, function_name):
    """Perform statistical analysis for MPSO replicates for every kind of value"""

    new_df_data = {
        "track_type": track_name,
        "mpso_type": mpso_name,
        "function_name": function_name
    }

    cols = list(df.columns)
    labels = ["time", "g_best_value", "time_ccd", "g_best_value_ccd"]
    for label in labels:
        if label in cols:
            new_df_data[f"average_{label}"] = df[label].mean()
            new_df_data[f"max_{label}"] = df[label].max()
            new_df_data[f"min_{label}"] = df[label].min()
            new_df_data[f"std-dev_{label}"] = df[label].std()

    return new_df_data

def run_benchmark_tests(
    replicates: int = 30, 
    seed: int = int(time.time()), 
    function_names: list[str] = [x for x in tf.TESTFUNCTSTRINGS if x not in IGNORELIST],
    mpso_types: list[int] = MPSO_TYPES,
    track_types: list[int] = TRACK_TYPES
    ):
    """ Run all benchmark tests.\n
    replicates:  Number of times to repeat each test
    seed:  random seed to use
    function_names:  List of test function names to use.
    mpso_types:  List of MPSO types (mpso and mpsoccd) to use.  mpso does not run ccd, mpsoccd does.
    track_types:  List of tracking types (time and quality) to use.  Time is optimized to use as few logging
    features as possible to speed up runtime and get an accurate time track.
    """
    np.random.seed(seed)

    # Make a base test folder
    test_path = _make_benchmark_folder()

    # Dump all of the settings we used into the base folder of the test
    with open(os.path.join(test_path, "settings.json"), "w+") as file:
        settings = {
            "replicates": replicates,
            "seed": seed,
            "functions": function_names,
            "mpso_types": [_return_mpso_type_name(mpso_type) for mpso_type in mpso_types],
            "track_types": [_return_track_type_name(track_type) for track_type in track_types]
        }
        json.dump(settings, file)

    # Array to store final MPSO replicate information
    results = []

    # Tracking variables.  Used to know how far along in the testing framework you are.
    total_test_size = len(track_types) * len(mpso_types) * len(function_names)
    test_number = 1

    # Make a directory to store the tests
    all_tests_path = os.path.join(test_path, "tests")
    os.mkdir(all_tests_path)
    for track_type, mpso_type, function_name in itertools.product(track_types, mpso_types, function_names):
        # Take the Cartesian product of all tracking types (time or quality), mpso types (mpso or mpsoccd)
        # And all of the functions (e.g. rosenbrock, sphere etc.).  Run a test for each one of these.
        extension_name = _build_extension_name(track_type, mpso_type, function_name)
        track_name, mpso_name = _get_test_names(track_type, mpso_type)

        # Make a new folder to store results for a specific test, and construct the MPSO logger.
        extension_path = _make_benchmark_test_folder(all_tests_path, extension_name)
        mpso_logger = _construct_MPSO(track_type=track_type, mpso_type=mpso_type, functionID = function_name)

        # Make a folder for each replicate
        individual_run_paths = os.path.join(extension_path, "MPSORuns")
        os.mkdir(individual_run_paths)

        print(f"Running {extension_name} (test {test_number} / {total_test_size})")

        mpso_rows = []
        # Run all replicates
        for i in range(replicates):
            start = time.time()
            print(f"\tReplicate {i+1}/{replicates} ... ", end="")
            mpso_logger.run_mpso(random_state = None)

            record_mpso_run(mpso_logger, individual_run_paths, i)
            # Get the final result
            end_result = mpso_logger.rows[-1]
            end_result["replicate_number"] = i+1
            # The below is useless information.
            if "mpso_iteration" in list(end_result.keys()):
                del end_result["mpso_iteration"]
            
            mpso_rows.append(end_result)
            print(f"Done, took {time.time()-start} seconds.")

        output_configs(mpso_logger, extension_path)
        output_csv_path = os.path.join(extension_path, "MPSORuns.csv")

        df = pd.DataFrame(mpso_rows)
        df.to_csv(output_csv_path)
        result = perform_statistical_analysis(df, track_name, mpso_name, function_name)
        pso_obj = mpso_logger.mpso.pso.pso.domain_data
        result["true bias"] = pso_obj.bias
        result["true optimum"] = u.np_to_json(pso_obj.optimum)

        results.append(result)
        test_number += 1
        
    pd.DataFrame(results).to_csv(os.path.join(test_path, "Results.csv"))

def _return_track_type_name(track_type: int):
    """Convert tracking type from integer to string"""
    if track_type == QUALITY:
        track_name = "quality"
    elif track_type == TIME:
        track_name = "time"
    return track_name

def _return_mpso_type_name(mpso_type: int):
    """Convert MPSO type from integer to string"""
    if mpso_type == MPSO:
        mpso_name = "mpso"
    elif mpso_type == MPSOCCD:
        mpso_name = "mpsoccd"
    return mpso_name

def _get_test_names(track_type, mpso_type):
    return _return_track_type_name(track_type), _return_mpso_type_name(mpso_type)

def _build_extension_name(track_type: int, mpso_type: int, functionID: str):
    """Build the specific extension name based on track type, mspo type, and function name"""
    track_name = _return_track_type_name(track_type)
    mpso_name = _return_mpso_type_name(mpso_type)
    extension_name = f"{mpso_name}-{track_name}-{functionID}"
    return extension_name

def _construct_MPSO(track_type: int, mpso_type: int, functionID):
    """Constructs an mpso logger based on preferences."""
    dataclasses = {}

    # Navigate to the config folder in benchmark test and load these configs.
    path_name = os.path.join(CONFIG_FOLDER, functionID) 
    config_names = os.listdir(path_name)
    for config_name in config_names:    
        with open(os.path.join(path_name, config_name)) as file:
            dataclasses[config_name[:-len(".json")]] = codec.json_file_to_dataclass(file)

    pso_logger_config: dc.PSOLoggerConfig = dataclasses["pso_logger_config"]
    mpso_logger_config: dc.MPSOLoggerConfig = dataclasses["mpso_logger_config"]
    mpso_config: dc.MPSOConfigs = dataclasses["mpso_config"]
    domain_data: dc.FunctionData = dataclasses["domain_data"]
    pso_hyperparameters: dc.PSOHyperparameters = dataclasses["pso_hyperparameters"]
    pso_config: dc.PSOConfig = dataclasses["pso_config"]
    ccd_hyperparameters: dc.CCDHyperparameters = dataclasses["ccd_hyperparameters"]
    
    if track_type == QUALITY:
        # If we are tracking quality, it is fine to track as many values as possible
        pso_logger_config.track_quality = True
        pso_logger_config.track_time = True
        mpso_logger_config.track_quality = True
        mpso_logger_config.track_time = True
        use_pso_logger = True
    elif track_type == TIME:
        # If we are tracking time, limit tracking to only MPSO iterations
        mpso_logger_config.track_quality = False
        mpso_logger_config.track_time = True
        use_pso_logger = False
    else:
        raise Exception(f"{track_type} Not a valid track type")
    
    # Adjust tracking configs based on whether we are using MPSO or MPSOCCD
    if mpso_type == MPSO:
        mpso_config.use_ccd = False
        mpso_logger_config.track_ccd = False
    elif mpso_type == MPSOCCD:
        mpso_config.use_ccd = True
        mpso_logger_config.track_ccd = True
    else:
        raise Exception(f"{mpso_type} not a valid run type")
    
    function = tf.TF.generate_function(functionID, optimum = domain_data.optimum, bias=domain_data.bias)
    
    pso = pso_file.PSO(
        pso_hyperparameters=pso_hyperparameters,
        domain_data = domain_data,
        function = function,
        pso_configs = pso_config
    )

    if use_pso_logger:
        pso = pso_file.PSOLogger(
            pso = pso,
            config = pso_logger_config
        )

    mpso = mpso_file.MPSO(
        pso=pso,
        mpso_config = mpso_config,
        ccd_hyperparameters = ccd_hyperparameters
    )

    mpso_logger = mpso_file.MPSOLogger(
        mpso = mpso,
        config = mpso_logger_config
    )

    return mpso_logger
