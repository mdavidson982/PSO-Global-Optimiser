import pso
import testfuncts
import numpy as np
import pandas as pd
import time
import utils.parameters as p
from datetime import datetime
import warnings
import json

warnings.filterwarnings("ignore", category=RuntimeWarning)

MPSORUN = 0
MPSOCCDRUN = 1
BENCHMARKFOLDER = "benchmarkruns"

IGNORELIST = [testfuncts.SHIFTEDELLIPTICID, 
              testfuncts.GRIEWANKID, 
              testfuncts.SHIFTEDROTATEDACKLEYID,
              testfuncts.ROTATEDRASTRIGINID
              ]


def record_run(runner: pso.MPSO_CCDRunner, mpso_or_mpsoccd: int, name: str, id: int):
    start = time.time()
    
    if mpso_or_mpsoccd == MPSORUN:
        print(f"Starting {name} mpso run")
        type = "MPSO"
        runner.mpso()
        print(f"Ended {name} mpso run")

    elif mpso_or_mpsoccd == MPSOCCDRUN:
        print(f"Starting {name} mpsoccd run")
        type = "MPSOCCD"
        runner.mpso_ccd()
        print(f"Ended {name} mpsoccd run")
    
    time_taken = time.time() - start
    pso_obj = runner.pso

def run_benchmark_tests():

    dims = 30
    dtype = np.float64
    
    upper_bound = np.ones(dims, dtype=dtype)*1000
    lower_bound = upper_bound*-1
    optimum = np.zeros(dims, dtype=dtype)
    current_datetime = datetime.now()


    for i in range(len(testfuncts.TESTFUNCTIDS)):
        name = testfuncts.TESTFUNCTSTRINGS[i]
        id = testfuncts.TESTFUNCTIDS[i]

        if name in IGNORELIST or id in IGNORELIST:
            continue

        pso_hyperparameters = PSOHyperparameters(
        num_part = p.NUM_PART,
        num_dim= p.NUM_DIM, 
        alpha = p.ALPHA,
        max_iterations=p.MAX_ITERATIONS, 
        w=p.W, 
        c1=p.C1, 
        c2=p.C2, 
        tolerance=p.TOLERANCE, 
        mv_iteration=p.NO_MOVEMENT_TERMINATION
        )

        ccd_hyperparameters = CCDHyperparameters(
            ccd_alpha=p.CCD_ALPHA, 
            ccd_tol=p.CCD_TOL, 
            ccd_max_its=p.CCD_MAX_ITS,
            ccd_third_term_its=p.CCD_THIRD_TERM_ITS
        )

        domain_data = DomainData(
            upper_bound = p.UPPER_BOUND,
            lower_bound = p.LOWER_BOUND
        )

        runner_config = MPSORunnerConfigs(use_ccd=True)

        optimum = optimum=p.OPTIMUM
        bias=p.BIAS,
        function = tf.TF.generate_function(p.FUNCT, optimum=optimum, bias=bias)

        pso = PSOData(
            pso_hyperparameters = pso_hyperparameters,
            ccd_hyperparameters = ccd_hyperparameters,
            domain_data = domain_data,
            function = function
        )

        logging_settings = PSOLoggerConfig(
            should_log=True
        )

        runner = MPSO_CCDRunner(
            pso=pso, 
            runs=5, 
            logging_settings=logging_settings,
            runner_settings=runner_config
        )
        runner = pso.MPSO_CCDRunner(pso_obj)

        for run_type in (MPSORUN, MPSOCCDRUN):
            rows.append(record_run(runner, run_type, name, id))
        
    print("Finished running test functions")


    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
    df = pd.DataFrame(rows, columns=columns)
    file_name = f"{BENCHMARKFOLDER}/MPSORESULTS_{formatted_datetime}.csv"

    print(f"Writing results to {file_name}")
    df.to_csv(file_name)

run_benchmark_tests()
    

        

        

        
    