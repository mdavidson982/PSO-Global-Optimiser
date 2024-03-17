from . import pso as pso_file, psodataclass as dc
from psofuncts import ccd

# Non-graphical runner
class MPSO:
    """
    Class for running MPSO
    """
    pso: pso_file.PSOInterface #Can either be a logging instance of PSO or a non-logging
    ccd_hyperparameters: dc.CCDHyperparameters
    runner_settings: dc.MPSORunnerConfigs

    def __init__(self, 
        pso: pso_file.PSOInterface,
        runner_settings: dc.MPSORunnerConfigs = dc.MPSORunnerConfigs(),
        ccd_hyperparameters: dc.CCDHyperparameters = None
    ):
        self.pso = pso
        self.runner_settings = runner_settings
        self.ccd_hypers = ccd_hyperparameters
        if not self.ccd_hypers.has_valid_learning_params():
            raise Exception("Bad learning parameters for CCD")

    def run_CCD(self):
        """Run CCD by taking g_best as a main input, and refining it"""
        ccd_hypers = self.ccd_hyperparameters
        g_best = ccd.CCD(
            initial=self.pso.pso.g_best, 
            lb = self.pso.pso.domain_data.lower_bound, 
            ub = self.pso.pso.domain_data.upper_bound,
            alpha = ccd_hypers.ccd_alpha, 
            tol = ccd_hypers.ccd_tol, 
            max_its = ccd_hypers.ccd_max_its,
            third_term_its = ccd_hypers.ccd_third_term_its, 
            func=self.pso.pso.function
        )
        self.pso.pso.g_best = g_best

    def mpso_ccd(self):
        """Runs PSO with CCD"""

        # Run MPSO
        for _ in range(self.runner_settings.iterations):
            self.run_PSO()
            if self.runner_settings.use_ccd:
                self.run_CCD()

class MPSOLogger:
    """Wrapper for the MPSO object, that enables logging of the run."""



