from . import pso as pso_file, psodataclass as dc
from psofuncts import ccd
import utils.parameters as p
from utils.util import np_to_json
import pandas as pd
import numpy as np
from time import time_ns

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
        runner_settings: dc.MPSOConfigs = dc.MPSOConfigs(),
        ccd_hyperparameters: dc.CCDHyperparameters = None
    ):
        self.pso = pso
        self.mpso_config = runner_settings
        self.ccd_hyperparameters = ccd_hyperparameters
        self.g_best = None
        self.iterations = 0

        if runner_settings.use_ccd:
            if ccd_hyperparameters is None:
                raise Exception("Configuration set to use ccd, but no hyperparameters supplied.")
            if not self.ccd_hyperparameters.has_valid_learning_params():
                raise Exception("Bad learning parameters for CCD")

    def run_CCD(self) -> None:
        """Run CCD by taking g_best from PSO as a main input, and refining it"""
        ccd_hypers = self.ccd_hyperparameters
        return ccd.CCD(
            initial=self.pso.pso.g_best, 
            lb = self.pso.pso.domain_data.lower_bound, 
            ub = self.pso.pso.domain_data.upper_bound,
            alpha = ccd_hypers.ccd_alpha, 
            tol = ccd_hypers.ccd_tol, 
            max_its = ccd_hypers.ccd_max_its,
            third_term_its = ccd_hypers.ccd_third_term_its, 
            func=self.pso.pso.function
        )

    def run_iteration(self) -> None:
        """
        Run a single iteration of PSO, optionally followed by CCD.
        temp_g_best is used as a starting point for the next pso instance, and will be
        overwritten by the result of CCD if CCD is enabled.
        """
        self.pso.run_PSO(self.g_best)
        self.g_best = self.pso.pso.g_best
        if self.mpso_config.use_ccd:
            self.g_best = self.run_CCD()
        self.iterations += 1
        
    def run_mpso(self) -> None:
        """Runs a full instance of MPSO, optionally with CCD"""
        while self.iterations < self.mpso_config.iterations:
            self.run_iteration()

class MPSOLogger:
    """Wrapper for the MPSO object, that enables logging of the run."""
    mpso: MPSO
    config: dc.MPSOLoggerConfig
    rows: list[dict] = {}

    def _check_PSOLogger(self):
        """Check to see if the underlying pso object is a logger"""
        return hasattr(self.mpso.pso, pso_file.PSOLogger.return_results.__name__)

    def return_results(self) -> pd.DataFrame:
        """Return the results of the logger"""
        return pd.DataFrame(self.rows)

    def run_iteration(self) -> None:
        """Runs an iteration of the underlying mpso object, and records any
        pertinent values.
        """
        pso_obj = self.mpso.pso.pso

        current_row = {}
        # Run an iteration of the underlying mpso object
        self.mpso.run_iteration()
        
        # Track quality of solution
        if self.config.track_quality:
            current_row.update({
                "mpso_iteration": self.mpso.iterations,
                "g_best_coords": np_to_json(pso_obj.get_g_best_coords()),
                "mpso_iteration": pso_obj.get_g_best_value(),
            })
        # Track time it took to get to the solution
        if self.config.track_time:
            current_row.update({
                "time": time_ns(),
            })

        # If the underlying pso object is a logger object, 
        # add this as a column.
        if self._check_PSOLogger():
            current_row.update({
                "PSODataframe": self.mpso.pso.return_results()
            })
        self.rows.append(current_row)

    def run_mpso(self) -> None:
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

def extension():

    z = pd.DataFrame(np.array(((1, 2, 3), (4, 5, 6))), columns=["this", "is", "a"])
    z2 = pd.DataFrame(np.array(((2.3, 4.9, 9.8), (5.1, -0.9, 3))))
    z3 = pd.DataFrame(np.array(((1.3, 4.9), (8.75, 4.9), (39.48, 8.7))))
    #print(z)

    v = pd.DataFrame({"idxs": [1, 2], "dfs": [z, z2]})
    v2 = pd.DataFrame({"idxs": [1, 2], "dfs": [z2, z3]})

    dorp = v
    v.to_json("./testeradf.json")
    v2.to_json("./testeradsf.json")

    t = pd.read_json("./testeradf.json")
    
    print(t)

    print(pd.DataFrame((t["dfs"].iloc[0])))
    print(pd.DataFrame((t["dfs"].iloc[1])))