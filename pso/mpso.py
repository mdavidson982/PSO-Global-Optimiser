from . import pso as pso_file, psodataclass as dc
from psofuncts import ccd
import utils.parameters as p
from utils.util import np_to_json
import pandas as pd
from time import time_ns
from numpy import random

# Non-graphical runner
class MPSO:
    """
    Class for running MPSO.

    pso:  PSO or PSOLogger that MPSO uses as a basis.
    mpso_config:  MPSO settings to be used for the running of MPSO.
    ccd_hyperparameters:  Determines the parameters used for CCD.  Should not be null if mpso_config is set to use CCD.

    g_best:  value used for 
    """
    pso: pso_file.PSOInterface #Can either be a logging instance of PSO or a non-logging
    mpso_config: dc.MPSOConfigs
    ccd_hyperparameters: dc.CCDHyperparameters
    iterations: int = 0
    g_best: p.ADTYPE = None

    def __init__(self, 
        pso: pso_file.PSOInterface,
        mpso_config: dc.MPSOConfigs = dc.MPSOConfigs(),
        ccd_hyperparameters: dc.CCDHyperparameters = None
    ):
        self.pso = pso
        self.mpso_config = mpso_config
        self.ccd_hyperparameters = ccd_hyperparameters
        self.initialize()

        if mpso_config.use_ccd:
            if ccd_hyperparameters is None:
                raise Exception("Configuration set to use ccd, but no hyperparameters supplied.")
            if not self.ccd_hyperparameters.has_valid_learning_params():
                raise Exception("Bad learning parameters for CCD")
            
    def initialize(self) -> None:
        self.g_best = None
        self.iterations = 0
            
    def run_PSO(self) -> None:
        self.pso.run_PSO(self.g_best, random_state = None)
        self.g_best = self.pso.pso.g_best

    def run_CCD(self) -> None:
        """Run CCD by taking g_best from PSO as a main input, and refining it"""
        ccd_hypers = self.ccd_hyperparameters
        self.g_best = ccd.CCD(
            initial=self.pso.pso.g_best, 
            lb = self.pso.pso.domain_data.lower_bound, 
            ub = self.pso.pso.domain_data.upper_bound,
            alpha = ccd_hypers.ccd_alpha, 
            tol = ccd_hypers.ccd_tol, 
            max_its = ccd_hypers.ccd_max_its,
            third_term_its = ccd_hypers.ccd_third_term_its, 
            func = self.pso.pso.function
        )

    def set_seed(self, random_state):
        if random_state >= 0:
            random.seed(random_state)
        elif random_state < 0:
            random.seed(self.mpso_config.seed)

    def run_iteration(self) -> None:
        """
        Run a single iteration of PSO, optionally followed by CCD.
        temp_g_best is used as a starting point for the next pso instance, and will be
        overwritten by the result of CCD if CCD is enabled.
        """
        self.run_PSO(self.g_best)
        if self.mpso_config.use_ccd:
            self.run_CCD()
        self.iterations += 1
        
    def run_mpso(self, random_state: int | None = None) -> None:
        self.set_seed(random_state)
        self.initialize()
        """Runs a full instance of MPSO, optionally with CCD"""
        while self.iterations < self.mpso_config.iterations:
            self.run_iteration()

class MPSOLogger:
    """Wrapper for the MPSO object, that enables logging of the run."""
    mpso: MPSO
    config: dc.MPSOLoggerConfig
    rows: list[dict] = []

    def __init__(self, mpso: MPSO, config: dc.MPSOLoggerConfig = dc.MPSOLoggerConfig()):
        self.mpso = mpso
        self.config = config

    def _check_PSOLogger(self):
        """Check to see if the underlying pso object is a logger"""
        return hasattr(self.mpso.pso, pso_file.PSOLogger.return_results.__name__)

    def return_results(self) -> pd.DataFrame:
        """Return the results of the logger as a dataframe"""
        return pd.DataFrame(self.rows)
    
    def write_results_to_json(self, filepath: str):
        """Write this specific dataframe to json"""
        self.return_results().to_json(filepath)

    def run_iteration(self) -> None:
        """Runs an iteration of the underlying mpso object, and records any
        pertinent values.
        """

        pso_obj = self.mpso.pso.pso
        current_row = {}
        # Run an iteration of the underlying mpso object
        self.mpso.run_PSO()

        # Track time it took to get to the solution
        if self.config.track_time:
            current_row.update({
                "time": time_ns(),
            })
        
        # Track quality of solution
        if self.config.track_quality:
            current_row.update({
                "mpso_iteration": self.mpso.iterations,
                "g_best_coords": np_to_json(pso_obj.get_g_best_coords()),
                "g_best_value": pso_obj.get_g_best_value(),
            })

        if self.mpso.mpso_config.use_ccd: 
            self.mpso.run_CCD()
            
            if self.config.track_ccd:
                if self.config.track_time:
                    current_row.update({
                        "time": time_ns()
                    })
                if self.config.track_quality:
                    current_row.update({
                        "g_best_coords_ccd": np_to_json(pso_obj.get_g_best_coords()),
                        "g_best_value_ccd": pso_obj.get_g_best_value()
                    })
            
        # If the underlying pso object is a logger object, 
        # add this as a column.
        if self._check_PSOLogger():
            current_row.update({
                "PSODataframe": self.mpso.pso.return_results()
            })
        self.rows.append(current_row)
        self.mpso.iterations += 1

    def run_mpso(self, random_state: int | None = None) -> None:
        self.mpso.set_seed(random_state)
        self.mpso.initialize()
        while self.mpso.iterations < self.mpso.mpso_config.iterations:
            self.run_iteration()

class MPSOInterface:

    def run_iteration(self) -> None:
        """
        Run a single iteration of PSO, optionally followed by CCD.
        temp_g_best is used as a starting point for the next pso instance, and will be
        overwritten by the result of CCD if CCD is enabled.
        """

    def run_mpso(self) -> None:
        """Runs a full instance of MPSO, optionally with CCD"""