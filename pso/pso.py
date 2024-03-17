from psofuncts.initializer import initializer
import psofuncts.update as up
import psodataclass as dc

import utils.util as u
import utils.parameters as p

import numpy as np
import pandas as pd
import time

#initialization variables
#calls the initialization function in the initializer.py. 
#Uses the parameter values created in parameters.py
#Returns: pos_matrix, an ndarray that keep the position of each particle (Initialized with random values for each dimmension)
#         vel_matrix, an ndarray that keeps the velocity for each particle (Initialized with random values for each dimmension)
#         p_best, an ndarray that keeps the personal minimum for each particle in each dimmension
#         g_best, an ndarray that keeps the global minimum between all particles in each dimmension
#         v_max, float based on the size of the area, this is the max velocity each particle can move 


#Here we will place the pso function. To achieve this we will loop through the velocity update, position update,
#p_best, and g_best functions until we get a satisfactory answer, and loop under the number of times defined in
#max iteration var

class PSO:
    """
    Holds the data that PSO uses to run its algorithm

    pso_hypers:         Hyperparameters for PSO (see psodataclass.PSOHyperparameters)
    ccd_hypers:         Hyperparameters for CCD (see psodataclass.CCDHyperparameters)
    domain_data:        Information about the domain of the objective function (see psodataclass.DomainData)
    pso_configs:        Contains additional configurations for PSO (see psodataclass.PSOConfig)

    function:           the function to be optimized
    pos_matrix:         matrix recording the position of the particles
    vel_matrix:         matrix recording the velocity of the particles
    p_best:             matrix recording the personal best of the particles
    g_best:             vector recording the global best of the instance

    seed:               Integer to use for RNG
    old_g_best:         matrix recording the last g_bests for use in second termination criteria
    iterations:         current iteration of the instance
    """
    pso_hypers: dc.PSOHyperparameters
    ccd_hypers: dc.CCDHyperparameters
    domain_data: dc.DomainData
    pso_configs: dc.PSOConfig
    function: any
    v_max: p.ADTYPE
    pos_matrix: p.ADTYPE
    vel_matrix: p.ADTYPE
    p_best: p.ADTYPE
    g_best: p.ADTYPE
    old_g_best: p.ADTYPE
    iteration: int = 0

    def __init__(self, 
        pso_hyperparameters: dc.PSOHyperparameters,
        ccd_hyperparameters: dc.CCDHyperparameters, 
        domain_data: dc.DomainData,
        function: any,
        pso_configs: dc.PSOConfig = dc.PSOConfig()
    ):
        
        np.random.seed(pso_configs.seed)

        self.pso_hypers = pso_hyperparameters
        if not self.pso_hypers.has_valid_learning_params():
            raise Exception("Bad learning parameters for PSO")
        
        self.ccd_hypers = ccd_hyperparameters
        if not self.ccd_hypers.has_valid_learning_params():
            raise Exception("Bad learning parameters for CCD")
        
        self.domain_data = domain_data

        self.pso_configs = pso_configs

        self.function = function
        self.g_best = None

    def initialize(self) -> None:
        """Run initialization to get necessary matrices"""
        pos_matrix, vel_matrix, p_best, g_best, v_max = initializer(
            num_part=self.pso_hypers.num_part, 
            num_dim=self.pso_hypers.num_dim, 
            alpha=self.pso_hypers.alpha, 
            upper_bound=self.domain_data.upper_bound,
            lower_bound=self.domain_data.lower_bound, 
            function = self.function
        )

        self.pos_matrix = pos_matrix
        self.vel_matrix = vel_matrix
        self.p_best = p_best
        self.v_max = v_max
        
        # If the previously recorded g_best from another run is better than the current g_best, keep it.
        # Otherwise, replace.
        if self.g_best is None or (g_best[-1] < self.g_best[-1]):
            self.g_best = g_best

        # Store the output value of past g_bests for the second terminating condition.  Multiplies a ones array by the maximum possible value
        # So that comparison starts out always being true.
        self.old_g_best = np.finfo(p.DTYPE).max*np.ones(self.pso_hypers.mv_iteration, dtype=p.DTYPE)
        self.iteration = 0

    def update(self) -> bool:
        """
        Performs an iteration of the PSO algorithm.  Returns whether the instance should terminate.
        """
        self.vel_matrix = up.update_velocity(
            v_part = self.vel_matrix, 
            x_pos = self.pos_matrix, 
            g_best = self.g_best, 
            p_best = self.p_best, 
            w = self.pso_hypers.w, 
            c1 = self.pso_hypers.c1, 
            c2 = self.pso_hypers.c2
        )
        
        self.vel_matrix = up.verify_bounds(
            upper_bounds = self.v_max, 
            lower_bounds = -self.v_max, 
            matrix = self.vel_matrix
        )
        
        self.pos_matrix = up.update_position(x_pos=self.pos_matrix, v_part=self.vel_matrix)

        self.pos_matrix = up.verify_bounds(
            upper_bounds = self.domain_data.upper_bound, 
            lower_bounds = self.domain_data.lower_bound, 
            matrix = self.pos_matrix
        )
        
        self.p_best = up.update_p_best(
            pos_matrix = self.pos_matrix, 
            past_p_best = self.p_best, 
            function = self.function
        )

        g_best = up.update_g_best(p_best=self.p_best)
        compare_g_bests = np.vstack((g_best, self.g_best))

        self.g_best = compare_g_bests[np.argmin(compare_g_bests[:, -1])]

        #roll simply shifts the numpy matrix over by 1.  So,
        # [1, 2, 3]
        # Where 1 might be the first image of g_best 2 might be the second image of g_best etc. becomes
        # [3, 1, 2]
        # We also want to store the image of the newest g_best.  This is done by replace what would be 3 in the above with the new image.
        # If 0 is the new image of g_best, then the above array becomes
        # [0, 1, 2] 
        self.old_g_best = np.roll(self.old_g_best, 1)
        self.old_g_best[0] = self.g_best[-1]

        self.iteration += 1

        return self.should_terminate()

    def should_terminate(self) -> bool:
        """Helper function which determines if the instance should terminate"""
        return self.iteration >= self.pso_hypers.max_iterations or self.second_termination()
    
    def second_termination(self) -> bool:
        """Second termination criteria, specified in document on page 7"""
        dividend = self.old_g_best[0]-self.old_g_best[-1]
        divisor = abs(self.old_g_best[0]) + self.pso_hypers.tolerance
        return abs(dividend/divisor) < self.pso_hypers.tolerance
    
    def set_g_best(self, g_best: p.ADTYPE) -> None:
        """Set the g_best for this object"""
        self.g_best = g_best

    def get_g_best_coords(self) -> p.ADTYPE:
        """Return the coordinates of the current gbest"""
        return self.g_best[:-1]
    
    def get_g_best_value(self) -> p.DTYPE:
        """Return the value of the current gbest"""
        return self.g_best[-1]
    
    def run_PSO(self):
        "Runs an instance of PSO"
        self.pso.initialize()
        shouldTerminate = False

        # pso.update() returns false when termination criteria have not been met,
        # and true when termination criteria have been met.
        while not shouldTerminate:
            shouldTerminate = self.pso.update()

    
    @property
    def pso(self):
        return self
    
class PSOLogger:
    """
    Wrapper for the PSO object, that enables logging of the run.
    """    
    pso: PSO = None
    config: dc.PSOLoggerConfig
    rows: list[dict] = []

    def __init__(self, 
        pso: PSO, 
        config: dc.PSOLoggerConfig = dc.PSOLoggerConfig()
    ):
        self.pso = pso
        self.config = config
        self.mpso_iterations = 0
        self.rows = []
        self.current_row = {}

    def record_row(self) -> None:
        if self.config.track_quality:
            self.current_row.update({
                "pso_iteration": self.pso.iteration,
                "g_best_coords": u.np_to_json(self.pso.get_g_best_coords()),
                "g_best_value": self.pso.get_g_best_value(),
            })
        if self.config.track_time:
            self.current_row.update({
                "time": time.time_ns(),
            })

    def add_current_row(self) -> None:
        self.rows.append(self.current_row)
        self.current_row = {}

    def return_results(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
    
    def initialize(self) -> None:
        self.mpso_iterations += 1
        self.pso.initialize()
        
    def update(self) -> bool:
        should_terminate = self.pso.update()
        self.record_row(is_ccd=False)
        return should_terminate

class PSOInterface:
    """
    Interface that defines PSO operations
    initialize():     Initialize the PSO data before the PSO loop
    update():         Update the PSO data according to PSO scheme
    CCD():            Run CCD from PSO data
    set_g_best():     Set the g_best of the PSO data
    """

    def initialize(self) -> None:
        """
        Run initialization to get necessary matrices
        """
        pass

    def update(self) -> bool:
        """
        Performs an iteration of the PSO algorithm.  Returns whether the instance should terminate.
        """
        pass

    def run_pso(self) ->

    @property
    def pso(self) -> PSO:
        pass
