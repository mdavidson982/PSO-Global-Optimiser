import pso
import testfuncts
import numpy as np
import pandas as pd
import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

MPSORUN = 0
MPSOCCDRUN = 1
BENCHMARKFOLDER = "benchmarkruns"

IGNORELIST = [testfuncts.SHIFTEDELLIPTICID, 
              testfuncts.GRIEWANKID, 
              testfuncts.SHIFTEDROTATEDACKLEYID,
              testfuncts.ROTATEDRASTRIGINID
              ]


def record_run(runner: pso.PSORunner, mpso_or_mpsoccd: int, name: str, id: int):
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

    return {
        "Function Name": name,
        "Function ID": id,
        "MPSO or MPSOCCD": type, 
        "Time taken": time_taken, 
        "Global best": pso_obj.g_best,
        #"Max iterations": pso_obj.iterations,
        "Dimensions": pso_obj.num_dim, 
        "Number of particles": pso_obj.num_part, 
        "Max Iterations": pso_obj.max_iterations, 
        "No Movement Termation": pso_obj.mv_iteration,
        "Tolerance": pso_obj.tolerance,
        "Upper Bound": pso_obj.upper_bound,
        "Lower Bound": pso_obj.lower_bound,
        "Optimum": pso_obj.optimum,
        "Bias": pso_obj.bias,
        "Alpha": pso_obj.alpha,
        "W": pso_obj.w,
        "C1": pso_obj.c1,
        "C2": pso_obj.c2,
        "CCD alpha": pso_obj.ccd_alpha,
        "CCD max iterations": pso_obj.ccd_max_its,
        "CCD tolerance": pso_obj.ccd_tol,
        "CCD third term its": pso_obj.ccd_third_term_its
    }

def run_benchmark_tests():

    dims = 30
    dtype = np.float64
    
    upper_bound = np.ones(dims, dtype=dtype)*1000
    lower_bound = upper_bound*-1
    optimum = np.zeros(dims, dtype=dtype)
    current_datetime = datetime.now()
    rows = []

    columns = ["Function Name",
                "Function ID",
                "MPSO or MPSOCCD" 
                "Time taken", 
                "Global best",
                #"Max iterations",
                "Dimensions", 
                "Number of particles", 
                "Max Iterations", 
                "No Movement Termation",
                "Tolerance",
                "Upper Bound",
                "Lower Bound",
                "Optimum",
                "Bias",
                "Alpha",
                "W",
                "C1",
                "C2",
                "CCD alpha",
                "CCD max iterations",
                "CCD tolerance",
                "CCD third term its"
                ]

    print("Running Test functions")
    for i in range(len(testfuncts.TESTFUNCTIDS)):
        name = testfuncts.TESTFUNCTSTRINGS[i]
        id = testfuncts.TESTFUNCTIDS[i]

        if name in IGNORELIST or id in IGNORELIST:
            continue

        pso_obj = pso.PSO(
            num_part = 50, 
            num_dim= dims, 
            alpha = 0.9, 
            upper_bound=upper_bound, 
            lower_bound=lower_bound,
            max_iterations=100, 
            w=0.8, 
            c1=0.4, 
            c2=0.4, 
            tolerance = 10**-6,
            mv_iteration= 20,
            optimum=optimum, 
            bias=0, 
            functionID=id, 
            mpso_runs=30,

            ccd_alpha=0.2, 
            ccd_tol=10**-6, 
            ccd_max_its=20,
            ccd_third_term_its=5
            )
        runner = pso.PSORunner(pso_obj)

        for run_type in (MPSORUN, MPSOCCDRUN):
            rows.append(record_run(runner, run_type, name, id))
        
    print("Finished running test functions")


    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M")
    df = pd.DataFrame(rows, columns=columns)
    file_name = f"{BENCHMARKFOLDER}/MPSORESULTS_{formatted_datetime}.csv"

    print(f"Writing results to {file_name}")
    df.to_csv(file_name)

run_benchmark_tests()
    

        

        

        
    