import time
from pso import pso as pso_file, mpso as mpso_file, psodataclass as dc, codec
import testfuncts.testfuncts as tf
import numpy as np
import pandas as pd
import os
import itertools
import json

from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

QUALITY = 0
TIME = 1

track_types = [QUALITY, TIME]

MPSO = 0
MPSOCCD = 1

mpso_types = [MPSO, MPSOCCD]

RESULTSFOLDER = os.path.join("benchmark_tests", "benchmarkruns")
CONFIG_FOLDER = os.path.join("benchmark_tests", "configs")

IGNORELIST = [tf.SHIFTEDELLIPTICSTRING, 
              tf.GRIEWANKSTRING, 
              tf.SHIFTEDROTATEDACKLEYSTRING,
              tf.ROTATEDRASTRIGINSTRING,
]

def _make_benchmark_folder():
    """Makes a run for the overall benchmark folder"""
    formatted_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    folder_path = os.path.join(RESULTSFOLDER, f"Benchmark{formatted_datetime}")
    os.mkdir(folder_path)
    return folder_path

def _make_benchmark_test_folder(benchmark_path: str, extension: str):
    """Makes a directory for every individual test"""
    extension_path = os.path.join(benchmark_path, extension)
    os.mkdir(extension_path)
    return extension_path

def _make_replicate_folder(benchmark_test_path: str, replicate: int):
    replicate_path = os.path.join(benchmark_test_path, f"MPSO-Iteration-{replicate}")
    os.mkdir(replicate_path)
    return replicate_path

def output_configs(logger: mpso_file.MPSOLogger, extension_path: str):
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
    if "PSOData" in list(logger.rows[-1].keys()):
        rows = []
        replicate_path = _make_replicate_folder(extension_path, iteration)
        pso_path = os.path.join(replicate_path, "PSOData.csv")

        for i, row in enumerate(logger.rows):
            pso_rows = row["PSOData"]
            for j in range(len(pso_rows)):
                pso_rows[j]["MPSOIteration"] = i+1
                pso_rows[j]["is_ccd"] = False
            rows.extend(pso_rows)

            if logger.config.track_ccd:
                ccd_row = {
                    "MPSOIteration": i+1,
                    "is_ccd":  True,
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

            del logger.rows[i]["PSOData"]
        pd.DataFrame(rows).to_csv(pso_path)

def perform_statistical_analysis(df: pd.DataFrame, track_name, mpso_name, function_name):
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

def run_benchmark_tests(replicates: int = 30, seed: int = int(time.time())):
    np.random.seed(seed)
    """ Run all benchmark tests """
    test_path = _make_benchmark_folder()

    results = []
    total_test_size = len(track_types) * len(mpso_types) * (len(tf.TESTFUNCTSTRINGS) - len(IGNORELIST))
    test_number = 1

    all_tests_path = os.path.join(test_path, "tests")
    os.mkdir(all_tests_path)
    for track_type, mpso_type, function_name in itertools.product(track_types, mpso_types, tf.TESTFUNCTSTRINGS):
        if function_name in IGNORELIST:
            continue
        extension_name = _build_extension_name(track_type, mpso_type, function_name)
        track_name, mpso_name = _get_test_names(track_type, mpso_type)

        extension_path = _make_benchmark_test_folder(all_tests_path, extension_name)
        mpso_logger = _construct_MPSO(track_type=track_type, mpso_type=mpso_type, functionID = function_name)

        print(f"Running {extension_name} (test {test_number} / {total_test_size})")

        mpso_rows = []
        for i in range(replicates):
            start = time.time()
            print(f"\tReplicate {i+1}/{replicates} ... ", end="")
            mpso_logger.run_mpso(random_state = None)
            record_mpso_run(mpso_logger, extension_path, i)
            end_result = mpso_logger.rows[-1]
            end_result["replicate_number"] = i+1
            del end_result
            
            mpso_rows.append(end_result)
            print(f"Done, took {time.time()-start} seconds.")

        output_configs(mpso_logger, extension_path)
        output_csv_path = os.path.join(extension_path, "MPSORuns.csv")

        df = pd.DataFrame(mpso_rows)
        df.to_csv(output_csv_path)

        results.append(perform_statistical_analysis(df, track_name, mpso_name, function_name))
        test_number += 1
    pd.DataFrame(results).to_csv(os.path.join(test_path, "Results.csv"))
    with open(os.path.join(test_path, "settings.json"), "w+") as file:
        json.dump({"replicates": replicates,"seed": seed}, file)

def _return_track_type_name(track_type: int):
    if track_type == QUALITY:
        track_name = "quality"
    elif track_type == TIME:
        track_name = "time"
    return track_name

def _return_mpso_type_name(mpso_type: int):
    if mpso_type == MPSO:
        mpso_name = "mpso"
    elif mpso_type == MPSOCCD:
        mpso_name = "mpsoccd"
    return mpso_name

def _get_test_names(track_type, mpso_type):
    return _return_track_type_name(track_type), _return_mpso_type_name(mpso_type)

def _build_extension_name(track_type: int, mpso_type: int, functionID: str):
    track_name = _return_track_type_name(track_type)
    mpso_name = _return_mpso_type_name(mpso_type)
    extension_name = f"{mpso_name}-{track_name}-{functionID}"
    return extension_name

def _construct_MPSO(track_type: int, mpso_type: int, functionID):
    # folder name for the benchmarktest
    dataclasses = {}
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
