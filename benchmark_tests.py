from pso import pso as pso_file, mpso as mpso_file, psodataclass as dc, codec
import testfuncts.testfuncts as tf
import utils.parameters as p
import numpy as np
import pandas as pd
import os
import time

from datetime import datetime
import warnings
import json

warnings.filterwarnings("ignore", category=RuntimeWarning)

MPSOQUALITY = 0
MPSOCCDQUALITY = 1
MPSOTIME = 2
MPSOCCDTIME = 3
run_types = [MPSOQUALITY, MPSOCCDQUALITY, MPSOTIME, MPSOCCDTIME]

BENCHMARKFOLDER = "benchmarkruns"

IGNORELIST = [tf.SHIFTEDELLIPTICID, 
              tf.GRIEWANKID, 
              tf.SHIFTEDROTATEDACKLEYID,
              tf.ROTATEDRASTRIGINID
              ]


def record_run(runner: mpso_file.MPSOLogger, name: str, mpso_runs: int = 30):
    pass

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

    pso_logger_config = dc.PSOLoggerConfig(
        log_level = dc.LogLevels.LOG_ALL,
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
        track_quality=True,
        track_time = True
    )

    return pso_hyperparameters, domain_data, pso_logger_config, ccd_hyperparameters, mpso_config, mpso_logger_config


def run_benchmark_tests():
    dims = 30
    dtype = np.float64
    optimum = np.zeros(dims, dtype=dtype)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
    
    folder_path = os.path.join(BENCHMARKFOLDER), f"BM{formatted_datetime}"

    os.mkdir(folder_path)

    pso_hyperparameters, domain_data, pso_logger_config, ccd_hyperparameters, mpso_config, mpso_logger_config = construct_configs(dims, dtype)

    for run_type in run_types:

        for i in range(len(tf.TESTFUNCTIDS)):
            name = tf.TESTFUNCTSTRINGS[i]
            id = tf.TESTFUNCTIDS[i]

            if name in IGNORELIST or id in IGNORELIST:
                continue

            if run_type == MPSOQUALITY:
                



            pso_config = dc.PSOConfig()

            function = tf.TF.generate_function(name, optimum=optimum, bias=0)

            pso = pso_file.PSO(
                pso_hyperparameters = pso_hyperparameters,
                domain_data = domain_data,
                function = function,
                pso_configs = pso_config
            )

            pso_logger = pso_file.PSOLogger(
                pso = pso,
                config = pso_logger_config
            )





            mpso = mpso_file.MPSO(
                pso = pso_logger,
                ccd_hyperparameters= ccd_hyperparameters,
                mpso_config = mpso_config
            )



            mpso_logger = mpso_file.MPSOLogger(
                mpso = mpso,
                mpso_logger_config = mpso_logger_config
            )

        for run_type in run_types:
            rows = record_run(mpso_logger, name=name, run_type=run_type, mpso_runs = 30)
        
    print("Finished running test functions")


    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
    df = pd.DataFrame(rows, columns=columns)
    file_name = f"{BENCHMARKFOLDER}/MPSORESULTS_{formatted_datetime}.csv"

    print(f"Writing results to {file_name}")
    df.to_csv(file_name)

run_benchmark_tests()