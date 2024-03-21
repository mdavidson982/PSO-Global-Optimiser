from pso import pso as pso_file, mpso as mpso_file, psodataclass as dc, codec
import testfuncts.testfuncts as tf
import utils.parameters as p
import numpy as np
import pandas as pd
import os
import logging

from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

QUALITY = 0
TIME = 1

track_types = [QUALITY, TIME]

MPSO = 0
MPSOCCD = 1

mpso_types = [MPSO, MPSOCCD]

BENCHMARKFOLDER = "benchmarkruns"

IGNORELIST = [tf.SHIFTEDELLIPTICID, 
              tf.GRIEWANKID, 
              tf.SHIFTEDROTATEDACKLEYID,
              tf.ROTATEDRASTRIGINID
              ]


def record_run(runner: mpso_file.MPSOLogger, func_name: str, mpso_runs: int = 30):
    rows = []
    for i in range(mpso_runs):
        runner.run_mpso()
        df = runner.return_results()
        
        row = {
            "df": df,
            "run_number": i+1,
            "function_name": func_name,
        }
        rows.append(row)
    return rows


def construct_configs(dims: int, dtype):
    pso_hyperparameters = dc.PSOHyperparameters(
        num_part = 50,
        num_dim = dims, 
        alpha = 0.7,
        max_iterations = 100, 
        w = 0.7, 
        c1 = 0.4, 
        c2 = 0.4, 
        tolerance=10**-6, 
        mv_iteration = 10
    )

    upper_bound = np.ones(dims, dtype=dtype)*100
    lower_bound = upper_bound*-1

    domain_data = dc.DomainData(
        upper_bound = upper_bound,
        lower_bound = lower_bound
    )

    pso_config = dc.PSOConfig()

    pso_logger_config = dc.PSOLoggerConfig(
        track_quality = True,
        track_time = True
    )

    ccd_hyperparameters = dc.CCDHyperparameters(
        ccd_alpha = 0.3, 
        ccd_tol = 10**-6, 
        ccd_max_its = 20,
        ccd_third_term_its = 4
    )

    mpso_config = dc.MPSOConfigs(
        use_ccd = True,
        iterations = 30
    )

    mpso_logger_config = dc.MPSOLoggerConfig(
        track_quality = True,
        track_time = True
    )

    return pso_hyperparameters, domain_data, pso_config, pso_logger_config, ccd_hyperparameters, mpso_config, mpso_logger_config

def run_benchmark_tests():
    dims = 30
    dtype = np.float64
    optimum = np.zeros(dims, dtype=dtype)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
    
    # Folder 
    folder_path = os.path.join(BENCHMARKFOLDER, f"Benchmark{formatted_datetime}")
    os.mkdir(folder_path)

    pso_hyperparameters, domain_data, pso_config, pso_logger_config, ccd_hyperparameters, mpso_config, mpso_logger_config = construct_configs(dims, dtype)
    use_pso_logger = None

    # Run through every type of tracker
    for run_type in [(x, y) for x in track_types for y in mpso_types]:
        track_type = run_type[0]
        mpso_type = run_type[1]

        # rows that will be included with the dataframe
        rows = []

        # Adjust tracking configs for whether we are doing something for quality or time
        if track_type == QUALITY:
            # If we are tracking quality, it is fine to track as many values as possible
            pso_logger_config.track_quality = True
            use_pso_logger = True
            track_name = "quality"# to be used for storing the results of benchmark tests
        elif track_type == TIME:
            # If we are tracking time, limit tracking to only MPSO iterations
            use_pso_logger = False
            track_name = "time" # to be used for storing the results of benchmark tests
        else:
            raise Exception(f"{track_type} Not a valid track type")

        # Adjust tracking configs based on whether we are using MPSO or MPSOCCD
        if mpso_type == MPSO:
            mpso_config.use_ccd = False
            mpso_logger_config.track_ccd = False
            mpso_name = "mpso" # to be used for storing the results of benchmark tests
        elif mpso_type == MPSOCCD:
            mpso_config.use_ccd = True
            mpso_logger_config.track_ccd = True
            mpso_name = "mpso_ccd" # to be used for storing the results of benchmark tests
        else:
            raise Exception(f"{mpso_type} not a valid run type")
        
        extension_name = f"{mpso_name}-{track_name}"
        logging.info(f"Running {extension_name} tests")
        
        # Run through all test functions
        for i in range(len(tf.TESTFUNCTIDS)):
            name = tf.TESTFUNCTSTRINGS[i]
            id = tf.TESTFUNCTIDS[i]

            if name in IGNORELIST or id in IGNORELIST:
                continue

            logging.info(f"Running {name} function")
            print(f"Running {name} function for {extension_name}")

            function = tf.TF.generate_function(id, optimum=optimum, bias=0)

            pso = pso_file.PSO(
                pso_hyperparameters = pso_hyperparameters,
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
                pso = pso,
                ccd_hyperparameters = ccd_hyperparameters,
                mpso_config = mpso_config
            )

            mpso_logger = mpso_file.MPSOLogger(
                mpso = mpso,
                config = mpso_logger_config
            )

            rows.extend(record_run(mpso_logger, func_name=name, mpso_runs = 30))

        # folder name for the benchmarktest
        dataclasses = [
            (pso_hyperparameters, "pso_hyperparameters"),
            (domain_data, "domain_data"),
            (pso_config, "pso_config"),
            (pso_logger_config, "pso_logger_config"),
            (ccd_hyperparameters, "ccd_hyperparameters"),
            (mpso_config, "mpso_config"),
            (mpso_logger_config, "mpso_logger_config")
        ]

        logging.info("Writing results to file")
        print("Writing results to file")

        extension_name = f"{mpso_name}-{track_name}"
        result_folder_path = os.path.join(folder_path, extension_name)
        os.mkdir(result_folder_path)

        with open(os.path.join(result_folder_path, "datadump.jsonl"), "w") as file:
            for row in pd.DataFrame(rows).iterrows():
                row[1].to_json(file)

        for dataclass in dataclasses:
            # Write the configs to a json file as well
            file_name = dataclass[1]
            data = dataclass[0]

            with open(os.path.join(result_folder_path, f"{file_name}.json"), "w") as file:
                codec.dataclass_to_json_file(data, file)

    print("Finished running test functions")

run_benchmark_tests()