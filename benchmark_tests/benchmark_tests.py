from pso import pso as pso_file, mpso as mpso_file, psodataclass as dc, codec
import testfuncts.testfuncts as tf
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
CONFIG_FOLDER = os.path.join("benchmark_tests", "configs")

IGNORELIST = [tf.SHIFTEDELLIPTICID, 
              tf.GRIEWANKID, 
              tf.SHIFTEDROTATEDACKLEYID,
              tf.ROTATEDRASTRIGINID
]

class BenchmarkTester:
    dims: int
    dtype: any
    folder_path: str


    def __init__(self, dims: int, dtype):
        self.dims = dims
        self.dtype = dtype

    def _make_benchmark_type_folder(self, extension_name: str):
        benchmark_type_folder = os.path.join(self.folder_path, extension_name)
        

    def _run_benchmark_objective(self, mpso_logger: mpso_file.MPSOLogger):


    def _run_benchmark_type(self, track_type: int, mpso_type: int):
        mpso_logger = construct_MPSO(track_type=track_type, mpso_type=mpso_type)
        if track_type == QUALITY:
            track_name = "quality"
        elif track_type == TIME:
            track_name = "time"
        
        if mpso_type == MPSO:
            mpso_name = "mpso"
        elif mpso_type == MPSOCCD:
            mpso_name = "mpsoccd"
        extension_name = f"{mpso_name}-{track_name}"

        logging.info(f"Running {extension_name} tests")
        self._make_benchmark_type_folder(extension_name)

        for i in range(len(tf.TESTFUNCTIDS)):
            name = tf.TESTFUNCTSTRINGS[i]
            id = tf.TESTFUNCTIDS[i]

            if name in IGNORELIST or id in IGNORELIST:
                continue

    def run_benchmark_tests(self):
        """ Run all benchmark tests """
        self._make_benchmark_folder()

        for run_type in [(x, y, z) for x in track_types for y in mpso_types for z in tf.TESTFUNCTSTRINGS]
            
            mpso_logger = construct_MPSO(track_type=track_type, mpso_type=mpso_type)
        
        for run_type in [(x, y) for x in track_types for y in mpso_types]:
            track_type = run_type[0]
            mpso_type = run_type[1]
            self._run_benchmark_type(track_type = track_type, mpso_type = mpso_type)

    def _make_benchmark_folder(self):
        """Makes a run for the overall benchmark folder"""
        formatted_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.folder_path = os.path.join(BENCHMARKFOLDER, f"Benchmark{formatted_datetime}")
        os.mkdir(self.folder_path)

    



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


def construct_MPSO(track_type: int, mpso_type: int):
    # folder name for the benchmarktest
    pso_hyperparameters = "pso_hyperparameters"
    domain_data = "domain_data"
    pso_config = "pso_config"
    pso_logger_config = "pso_logger_config"
    ccd_hyperparameters = "ccd_hyperparameters"
    mpso_config = "mpso_config"
    mpso_logger_config = "mpso_logger_config"

    dataclass_file_names = [
        pso_hyperparameters,
        domain_data,
        pso_config,
        pso_logger_config,
        ccd_hyperparameters,
        mpso_config,
        mpso_logger_config
    ]

    dataclasses = {}

    for file_name in dataclass_file_names:
        with open(os.path.join(CONFIG_FOLDER, file_name+".json")) as file:
            dataclasses[file_name] = [codec.json_file_to_dataclass(file)]

    with open(os.path.join(CONFIG_FOLDER, "pso_hyperparameters")) as file:
        pso_hyperparameters = codec.json_file_to_dataclass(file)
    
    if track_type == QUALITY:
        # If we are tracking quality, it is fine to track as many values as possible
        dataclasses[pso_logger_config].track_quality = True
        dataclasses[pso_logger_config].track_time = True
        dataclasses[mpso_logger_config].track_quality = True
        dataclasses[mpso_logger_config].track_time = True
        use_pso_logger = True
    elif track_type == TIME:
        # If we are tracking time, limit tracking to only MPSO iterations
        dataclasses[mpso_logger_config].track_quality = False
        dataclasses[mpso_logger_config].track_time = True
        use_pso_logger = False
    else:
        raise Exception(f"{track_type} Not a valid track type")
    
    # Adjust tracking configs based on whether we are using MPSO or MPSOCCD
    if mpso_type == MPSO:
        dataclasses[mpso_config].use_ccd = False
        mpso_logger_config.track_ccd = False
    elif mpso_type == MPSOCCD:
        mpso_config.use_ccd = True
        mpso_logger_config.track_ccd = True
    else:
        raise Exception(f"{mpso_type} not a valid run type")
    
    pso = pso_file.PSO(
        pso_hyperparameters=dataclasses[pso_hyperparameters],
        domain_data = dataclasses[domain_data],
        function = None,
        pso_configs = dataclasses[pso_config]
    )

    if use_pso_logger:
        pso = pso_file.PSOLogger(
            pso = pso,
            config = dataclasses[pso_logger_config]
        )

    mpso = mpso_file.MPSO(
        pso=pso,
        mpso_config = dataclasses[mpso_config],
        ccd_hyperparameters=dataclasses[ccd_hyperparameters]
    )

    mpso_logger = mpso_file.MPSOLogger(
        mpso = mpso,
        config = dataclasses[mpso_logger_config]
    )

    return mpso_logger



def run_benchmark_objective_function():
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

def run_benchmark_type(track_type: int, mpso_type: int, folder_path: str, dims: int, dtype):
    """ Run a specific type of benchmark (e.g. mpso_ccd-quality, mpso_ccd-time etc.)"""
    configs = construct_configs(dims, dtype)
    pso_hyperparameters = configs[0]
    domain_data = configs[1]
    pso_config = configs[2]
    pso_logger_config = configs[3]
    ccd_hyperparameters = configs[4]
    mpso_config = configs[5]
    mpso_logger_config = configs[6]
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
        run_benchmark_objective_function()
    

def run_benchmark_tests():
    dims = 30
    dtype = np.float64
    optimum = np.zeros(dims, dtype=dtype)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H-%M-%S")
    
    # Folder 
    folder_path = os.path.join(BENCHMARKFOLDER, f"Benchmark{formatted_datetime}")
    os.mkdir(folder_path)

    # Run through every type of tracker
    for run_type in [(x, y) for x in track_types for y in mpso_types]:
        track_type = run_type[0]
        mpso_type = run_type[1]
        run_benchmark_type(track_type = track_type, mpso_type = mpso_type, folder_path = folder_path)

        # rows that will be included with the dataframe
        rows = []

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