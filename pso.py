import numpy as np
import initializer as ini
import parameters as p
import update as up
import testfuncts as tf
import ccd
import time
import pandas as pd
from dataclasses import dataclass

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

class PSOInterface:
    """
    Interface that defines PSO operations

    initialize():     Initialize the PSO data before the PSO loop
    update():         Update the PSO data according to PSO scheme
    CCD():            Run CCD from PSO data
    set_g_best():     Set the g_best of the PSO data
    """

    g_best: p.ADTYPE
    
    def initialize(self):
        """
        Run initialization to get necessary matrices
        """
        pass

    def update(self) -> bool:
        """
        Performs an iteration of the PSO algorithm.  Returns whether the instance should terminate.
        """
        pass
    
    def CCD(self) -> p.ADTYPE:
        """
        Run CCD by taking g_best as a main input, and refining it
        """
        pass

    

@dataclass        
class PSOHyperparameters:
    """Class which holds PSO hyperparameters

    num_part:           number of particles the instance will use
    num_dim:            dimensions of the problem to be solved
    alpha:              velocity constriction parameter
    max_iterations:     maximum number of iterations the instance can run through
    w:                  momentum parameter (see docs for more)
    c1:                 cognitive parameter (see docs for more)
    c2:                 social parameter (see docs for more)
    tolerance:          part of second termination criteria (see docs)
    mv_iteration:       part of second termination criteria (see docs)

    has_valid_learning_params():  Determines if the parameters are valid
    """

    num_part: int
    num_dim: int
    alpha: p.ADTYPE
    w: p.ADTYPE
    c1: p.ADTYPE
    c2: p.ADTYPE
    tolerance: p.ADTYPE
    mv_iteration: int
    max_iterations: int

    def has_valid_learning_params(self) -> bool:
        if not (self.alpha < 1 and self.alpha > 0):
            return False
        if not (self.tolerance > 0):
            return False
        if not (self.num_part > 0):
            return False
        if not (self.num_dim > 0):
            return False
        if not (self.mv_iteration > 0):
            return False
        if not (self.max_iterations > 0):
            return False
        if not (self.c1 < 2 and self.c1 > 0):
            return False
        if not (self.c2 < 2 and self.c2 > 0):
            return False
        # See page 5 of manuscript for below condition
        if not (self.w < 1 and self.w > 0.5*(self.c1+self.c2)-1):
            return False
        return True
    
@dataclass    
class CCDHyperparameters:
    """
    ccd_alpha:          Restricts the domain for ccd
    ccd_max_its:        Number of max iterations for ccd
    ccd_tol:            Tolerance parameter to determine if ccd is improving or not improving
    ccd_third_term_its: Number of iterations to determine if tolerance threshold reached
    
    has_valid_learning_parms():  Determines if the parameters are valid
    """

    ccd_alpha: p.DTYPE
    ccd_max_its: int
    ccd_tol: p.DTYPE
    ccd_third_term_its: int

    def has_valid_learning_params(self) -> bool:
        if not (self.ccd_alpha < 1 and self.ccd_alpha > 0):
            return False
        if not (self.ccd_max_its > 0):
            return False
        if not (self.ccd_tol > 0):
            return False
        if not (self.ccd_third_term_its > 0):
            return False
        return True
    

class DomainData:
    """
    Class that holds information about the objective function's
    domain for PSO

    upper_bound:        upper bounds of the domain
    lower_bound:        lower bounds of the domain
    v_max:              vector controlling the maximum velocity of particles
    """
    upper_bound: p.ADTYPE
    lower_bound: p.ADTYPE
    v_max: p.ADTYPE

    def __init__(self, upper_bound: p.ADTYPE, lower_bound: p.ADTYPE):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

@dataclass
class IterationData:
    """
    Class that holds information about a specific iteration of PSO

    iteration_num:      iteration of pso
    pos_matrix:         matrix recording the position of the particles
    vel_matrix:         matrix recording the velocity of the particles
    p_best:             matrix recording the personal best of the particles
    recorded_g_best:    vector of the g_best that was stored in PSO.  Because of how
                        MPSO works, it may be that the g_best that an iteration found is not
                        better than a previous run of PSO.
    iteration_g_best:   Opposite of recorded_g_best, records the g_best that this particular
                        instance found before comparison with other iterations.

    old_g_bests:        All of the old g_bests that were recorded up to this point.
    """
    iteration_num: int
    pos_matrix: p.ADTYPE
    vel_matrix: p.ADTYPE
    p_best: p.ADTYPE
    recorded_g_best: p.ADTYPE
    iteration_g_best: p.ADTYPE
    old_g_bests: p.ADTYPE

@dataclass
class PSOData:
    """
    Holds the data that PSO uses to run its algorithm

    function:           the function to be optimized
    pos_matrix:         matrix recording the position of the particles
    vel_matrix:         matrix recording the velocity of the particles
    p_best:             matrix recording the personal best of the particles
    g_best:             vector recording the global best of the instance

    seed:               Integer to use for RNG
    old_g_best:         matrix recording the last [mv_iteration] g_bests for use in second termination criteria
    iterations:         current iteration of the instance



    """
    pso_hypers: PSOHyperparameters
    ccd_hypers: CCDHyperparameters
    domain_data: DomainData
    mv_iteration: int
    function: any
    pos_matrix: p.ADTYPE
    vel_matrix: p.ADTYPE
    p_best: p.ADTYPE
    g_best: p.ADTYPE
    old_g_best: p.ADTYPE
    seed: int
    iterations: int = 0

    def __init__(self, pso_hyperparameters: PSOHyperparameters,
                ccd_hyperparameters: CCDHyperparameters, domain_data: DomainData,
                function: any, seed: int = int(time.time())):
        
        np.random.seed(seed)

        self.pso_hypers = pso_hyperparameters
        if not self.pso_hypers.has_valid_learning_params():
            raise Exception("Bad learning parameters for PSO")
        
        self.ccd_hypers = ccd_hyperparameters
        if not self.ccd_hypers.has_valid_learning_params():
            raise Exception("Bad learning parameters for CCD")
        
        self.domain_data = domain_data

        self.function = function
        self.seed = seed
        self.g_best = None

    def initialize(self):
        """Run initialization to get necessary matrices"""
        pos_matrix, vel_matrix, p_best, g_best, v_max = ini.initializer(
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
        self.domain_data.v_max = v_max
        
        # If the previously recorded g_best from another run is better than the current g_best, keep it.
        # Otherwise, replace.
        if self.g_best is None or (g_best[-1] < self.g_best[-1]):
            self.g_best = g_best

        # Store the output value of past g_bests for the second terminating condition.  Multiplies a ones array by the maximum possible value
        # So that comparison starts out always being true.
        self.old_g_best = np.finfo(p.DTYPE).max*np.ones(self.pso_hypers.mv_iteration, dtype=p.DTYPE)
        self.iterations = 0

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
            upper_bounds = self.domain_data.v_max, 
            lower_bounds = -self.domain_data.v_max, 
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

        self.iterations += 1

        return self.should_terminate()
    
    def CCD(self) -> p.ADTYPE:
        """Run CCD by taking g_best as a main input, and refining it"""
        ccd_hypers = self.ccd_hypers
        return (ccd.CCD(
                initial=self.g_best, 
                lb = self.domain_data.lower_bound, 
                ub = self.domain_data.upper_bound,
                alpha = ccd_hypers.ccd_alpha, 
                tol = ccd_hypers.ccd_tol, 
                max_its = ccd_hypers.ccd_max_its,
                third_term_its = ccd_hypers.ccd_third_term_its, 
                func=self.function)
        )

    def should_terminate(self) -> bool:
        """Helper function which determines if the instance should terminate"""
        return self.iterations >= self.pso_hypers.max_iterations or self.second_termination()
    
    def second_termination(self) -> bool:
        """Second termination criteria, specified in document on page 7"""
        dividend = self.old_g_best[0]-self.old_g_best[-1]
        divisor = abs(self.old_g_best[0]) + self.pso_hypers.tolerance
        return abs(dividend/divisor) < self.pso_hypers.tolerance
    
    def set_g_best(self, g_best: p.ADTYPE) -> None:
        """Set the g_best for this object"""
        self.g_best = g_best
    
@dataclass
class PSOLoggerConfig:
    """
    Settings for the PSOLogger class.
    """
    should_log: bool = False

class PSOLogger:
    """
    Wrapper for the PSO object, that enables logging.
    """    
    pso: PSOData = None
    config: PSOLoggerConfig
    mpso_iterations: int = 0

    def __init__(self, pso: PSOData, config: PSOLoggerConfig = PSOLoggerConfig()):
        self.pso = pso
        self.config = config
        self.mpso_iterations = 0
        self.df = pd.DataFrame()
    
    def initialize(self):
        self.mpso_iterations += 1
        self.pso.initialize()

    def update(self) -> bool:
        should_terminate = self.pso.update()
        return should_terminate
    
    def CCD(self) -> p.ADTYPE:
        return self.pso.CCD()
    
    @property
    def g_best(self):
        """Function to enable calling g_best directly"""
        return self.pso.g_best
    
    @g_best.setter
    def g_best(self, new_g_best):
        self.pso.g_best = new_g_best

@dataclass
class MPSORunnerConfigs:
    """Class which defines the configurations for the MPSO Runner"""
    use_ccd: bool = True
        
# Non-graphical runner
class MPSO_CCDRunner:
    "Runner class.  Can run PSO, MPSO, or MPSO-CCD."
    pso: PSOInterface #Can either be a logging instance of PSO or a non-logging
    runner_settings: MPSORunnerConfigs
    runs: int

    def __init__(self, pso: PSOData, runs: int = 30, logging_settings: PSOLoggerConfig = PSOLoggerConfig(),
                 runner_settings: MPSORunnerConfigs = MPSORunnerConfigs()):
        if logging_settings.should_log:
            self.pso = PSOLogger(pso=pso, config=logging_settings)
        else:
            self.pso = pso

        self.runner_settings = runner_settings
        self.runs = runs

    def run_PSO(self):
        "Runs PSO"
        self.pso.initialize()
        shouldTerminate = False

        # pso.update() returns false when termination criteria have not been met,
        # and true when termination criteria have been met
        while not shouldTerminate:
            shouldTerminate = self.pso.update()

    def mpso_ccd(self):
        """Runs PSO with CCD"""
        start = time.time()

        for _ in range(self.runs):
            self.run_PSO()
            if self.runner_settings.use_ccd:
                refined_g_best = self.pso.CCD()
                self.pso.g_best = refined_g_best
            
        print(f"it took {time.time() - start} seconds to run")

def test_PSO():

    pso_hyperparameters = PSOHyperparameters(
        num_part = p.NUM_PART,
        num_dim=p.NUM_DIM, 
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

    runner.mpso_ccd()
    #runner = PSORunner(pso)
    #runner.mpso_ccd()        """Set the g_best for this object"""
    print(runner.pso.g_best)

    

test_PSO()





    #terminating conditions: will either terminate when
    #1. max iterations reached
    #2. minimal movements after a few iterations. If the g_best doesn't move much and fits within our tolerance, we
    # exit




