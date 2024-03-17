"""
TO RUN:
Make sure workspace is in root directory (PSO-Global-Optimiser)
python3 -m unittest pso/test/test_pso.py
"""

from .. import pso as pso_file, psodataclass as dc
import numpy as np
import unittest 
import testfuncts.testfuncts as tf
import logging

class PSOTester(unittest.TestCase):

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

        domain_data = dc.DomainData(
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
    
    def construct_logger(self):
        pso = self.construct_pso()
        logging_settings = dc.PSOLoggerConfig(
            log_level = dc.LogLevels.NO_LOG
        )

        pso_logger = pso_file.PSOLogger(
            pso = pso,
            config = logging_settings
        )
        return pso_logger

    def test_pso_initialize(self):
        pso = self.construct_pso()
        g_best = np.random.rand(pso.pso_hypers.num_dim + 1) 

        pso.initialize(start_g_best=g_best)
        np.testing.assert_array_almost_equal(g_best, pso.g_best, 8)

    def test_pso_run(self):
        pso = self.construct_pso()
        pso.run_PSO()
        logging.info(pso.iteration)
        logging.info(pso.g_best)

    def test_logger_run(self):
        logger = self.construct_logger()
        logger.run_PSO()