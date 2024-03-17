from . import pso as pso_file, psodataclass as dc
from psofuncts import ccd
import utils.parameters as p

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
        
    def run_mpso(self) -> None:
        """Runs a full instance of MPSO, optionally with CCD"""
        for _ in range(self.mpso_config.iterations):
            self.run_iteration()

class MPSOLogger:
    mpso: MPSO

    """Wrapper for the MPSO object, that enables logging of the run."""

class MPSOInterface:
    """Interface which defines the functions for MPSO"""
    def run_CCD(self) -> None:
        """Run CCD by taking g_best from PSO as a main input, and refining it"""

    def run_iteration(self) -> None:
        """
        Run a single iteration of PSO, optionally followed by CCD.
        temp_g_best is used as a starting point for the next pso instance, and will be
        overwritten by the result of CCD if CCD is enabled.
        """

    def run_mpso(self) -> None:
        """Runs a full instance of MPSO, optionally with CCD"""