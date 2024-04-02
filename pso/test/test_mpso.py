"""
TO RUN:
Make sure workspace is in root directory (PSO-Global-Optimiser)
python3 -m unittest pso/test/test_mpso.py
"""

from .. import pso as pso_file, psodataclass as dc, mpso as mpso_file
import numpy as np
import unittest 
import testfuncts.testfuncts as tf
import logging

class MPSOTester(unittest.TestCase):

    logging.basicConfig(level=logging.INFO)

    def construct_pso(self) -> pso_file.PSO:
        num_dim = 5

        pso_hyperparameters = dc.PSOHyperparameters(
            num_part = 5,
            num_dim = num_dim, 
            alpha = 0.7,
            max_iterations=100, 
            w=0.75, 
            c1=0.3, 
            c2=0.3, 
            tolerance = 10**-6, 
            mv_iteration = 10
        )

        domain_data = dc.FunctionData(
            upper_bound = np.ones(num_dim)*100,
            lower_bound = np.ones(num_dim)*-100
        )

        optimum = np.zeros(num_dim)
        bias = 0
        
        function = tf.TF.generate_function("rosenbrock", optimum=optimum, bias=bias)

        pso = pso_file.PSO(
            pso_hyperparameters = pso_hyperparameters,
            domain_data = domain_data,
            function = function
        )
        return pso
    
    def construct_pso_logger(self):
        pso = self.construct_pso()
        logging_settings = dc.PSOLoggerConfig(
            log_level = dc.LogLevels.NO_LOG
        )

        pso_logger = pso_file.PSOLogger(
            pso = pso,
            config = logging_settings
        )
        return pso_logger
    
    def construct_mpso(self, pso_logging: bool, use_ccd: bool):
        if pso_logging:
            pso = self.construct_pso_logger()
        else:
            pso = self.construct_pso()
        
        runner_settings = dc.MPSOConfigs(
            use_ccd = use_ccd,
            iterations = 30
        )

        ccd_hyperparameters = dc.CCDHyperparameters(
            ccd_alpha=0.2,
            ccd_max_its = 20,
            ccd_tol = 10**-6,
            ccd_third_term_its=5
        )

        mpso = mpso_file.MPSO(
            pso=pso,
            mpso_config=runner_settings,
            ccd_hyperparameters=ccd_hyperparameters
        )

        return mpso
    
    def construct_mpso_logger(self, pso_logging: bool, use_ccd: bool):
        mpso = self.construct_mpso(pso_logging, use_ccd)
        mpso_logger = mpso_file.MPSOLogger()

    def test_mpso_iteration(self):
        mpso = self.construct_mpso(False, False)
        mpso.run_iteration()
        logging.info("MPSO iteration without CCD")
        logging.info(mpso.pso.pso.iteration)
        logging.info(f"{mpso.g_best} \n")

    def test_mpso_ccd_iteration(self):
        mpso = self.construct_mpso(False, True)
        mpso.run_iteration()
        logging.info("MPSO iteration With CCD")
        logging.info(mpso.pso.pso.iteration)
        logging.info(f"{mpso.g_best} \n")

    def test_mpso(self):
        mpso = self.construct_mpso(False, False)
        mpso.run_mpso()
        logging.info("MPSO full run without CCD")
        logging.info(mpso.pso.pso.iteration)
        logging.info(f"{mpso.g_best} \n")

    def test_mpso_with_ccd(self):
        mpso = self.construct_mpso(False, True)
        mpso.run_mpso()
        logging.info("MPSO full run with CCD")
        logging.info(mpso.pso.pso.iteration)
        logging.info(f"{mpso.g_best} \n")