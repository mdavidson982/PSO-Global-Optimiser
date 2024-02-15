import numpy as np
import initializer as ini
import parameters as p
import update as up
import testfuncts as tf
import ccd
import time

def validate_learn_params(w: p.DTYPE, c1: p.DTYPE, c2: p.DTYPE) -> bool:
    """Determines if learning params are correct"""
    return w < 1 and w > 0.5*(c1+c2)

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
    Controls the operation of the PSO algorithm.  

    num_part:           number of particles the instance will use
    num_dim:            dimensions of the problem to be solved
    alpha:              velocity constriction parameter
    upper_bound:        upper bounds of the domain
    lower_bound:        lower bounds of the domain
    max_iterations:     maximum number of iterations the instance can run through
    w:                  momentum parameter (see docs for more)
    c1:                 cognitive parameter (see docs for more)
    c2:                 social parameter (see docs for more)
    tolerance:          part of second termination criteria (see docs)
    mv_iteration:       part of second termination criteria (see docs)
    function:           the function to be optimized
    pos_matrix:         matrix recording the position of the particles
    vel_matrix:         matrix recording the velocity of the particles
    p_best:             matrix recording the personal best of the particles
    g_best:             vector recording the global best of the instance
    v_max:              vector controlling the maximum velocity of particles
    old_g_best:         matrix recording the last [mv_iteration] g_bests for use in second termination criteria
    iterations:         current iteration of the instance
    optimum:            shifted optimum of the function
    bias:               bias of the function
    functionID:         which function to use

    ccd_alpha:          Restricts the domain for ccd
    ccd_max_its:        Number of max iterations for ccd
    ccd_tol:            Tolerance parameter to determine if ccd is improving or not improving
    ccd_third_term_its: Number of iterations to determine if tolerance threshold reached

    """
    num_part: int 
    num_dim: int
    alpha: p.DTYPE
    upper_bound: p.ADTYPE
    lower_bound: p.ADTYPE
    max_iterations: int
    w: p.DTYPE
    c1: p.DTYPE
    c2: p.DTYPE
    tolerance: p.DTYPE
    mv_iteration: int
    function: any
    pos_matrix: p.ADTYPE
    vel_matrix: p.ADTYPE
    p_best: p.ADTYPE
    g_best: p.ADTYPE
    v_max: p.ADTYPE
    old_g_best: p.ADTYPE
    iterations: int = 0
    optimum: p.ADTYPE
    bias: p.DTYPE
    mpso_runs: int

    ccd_alpha: p.DTYPE
    ccd_max_its: int
    ccd_tol: p.DTYPE
    ccd_third_term_its: int

    functionID: int | str

    def __init__(self, num_part: int, num_dim: int, alpha: p.DTYPE, 
                upper_bound: p.ADTYPE, lower_bound: p.ADTYPE,
                max_iterations: int, w: p.DTYPE, c1: p.DTYPE, c2: p.DTYPE, tolerance: p.DTYPE,
                mv_iteration: int, optimum: p.ADTYPE, bias: p.DTYPE, functionID: str | int, mpso_runs: int,
                ccd_alpha: p.DTYPE, ccd_max_its: int, ccd_tol: p.DTYPE, ccd_third_term_its: int):
        self.num_part = num_part
        self.num_dim = num_dim
        self.alpha = alpha
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tolerance = tolerance
        self.mv_iteration = mv_iteration
        self.optimum = optimum
        self.bias = bias
        self.functionID = functionID
        self.mpso_runs = mpso_runs
        self.function = tf.TF.generate_function(functionID=functionID, optimum=optimum, bias=bias)
        self.g_best = None

        self.ccd_alpha = ccd_alpha
        self.ccd_max_its = ccd_max_its
        self.ccd_tol = ccd_tol
        self.ccd_third_term_its = ccd_third_term_its

        if not validate_learn_params(w, c1, c2):
            raise Exception("Bad learning parameters")

    def initialize(self):
        # Run initialization to get necessary matrices
        self.pos_matrix, self.vel_matrix, self.p_best, g_best, self.v_max = ini.initializer(
            num_part=self.num_part, num_dim=self.num_dim, alpha=self.alpha, upper_bound=self.upper_bound,
            lower_bound=self.lower_bound, function = self.function)
        
        # If the previously recorded g_best from another run is better than the current g_best, keep it.
        # Otherwise, replace.

        
        if self.g_best is None or (g_best[-1] < self.g_best[-1]):
            self.g_best = g_best

        # Store the image of past g_bests for the second terminating condition.  Multiplies a ones array by the maximum possible value
        # So that comparison starts out always being true.
        self.old_g_best = np.finfo(p.DTYPE).max*np.ones(self.mv_iteration, dtype=p.DTYPE)
        self.iterations = 0

    def update(self) -> bool:
        """
        Performs an iteration of the PSO algorithm.  Returns whether the instance should terminate.
        """
        self.vel_matrix = up.update_velocity(self.vel_matrix, self.pos_matrix, self.g_best, self.p_best, self.w, self.c1, self.c2)
        self.vel_matrix = up.verify_bounds(self.v_max, -self.v_max, self.vel_matrix)
        self.pos_matrix = up.update_position(self.pos_matrix, self.vel_matrix)
        #added verify bound to the pso loop. Assumed position matrix was the correct input. Putting this comment here to make sure that's right later when we review.
        self.pos_matrix = up.verify_bounds(upper_bound = self.upper_bound, lower_bound = self.lower_bound, matrix = self.pos_matrix)
        self.p_best = up.update_p_best(pos_matrix= self.pos_matrix, past_p_best = self.p_best, function = self.function)

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

    def should_terminate(self) -> bool:
        """Helper function which determines if the instance should terminate"""
        return self.iterations >= self.max_iterations or self.second_termination()
    
    def second_termination(self) -> bool:
        """Second termination criteria, specified in document"""
        return (abs(self.old_g_best[0]-self.old_g_best[-1])/(abs(self.old_g_best[0]) + self.tolerance)) < self.tolerance

# Non-graphical runner
class PSORunner:
    "Runs an instance of the PSO algorithm."
    pso: PSO

    def __init__(self, pso: PSO):
        self.pso = pso

    def run_PSO(self):
        "Runs PSO"
        self.pso.initialize()
        shouldTerminate = False
        while not shouldTerminate:
            shouldTerminate = self.pso.update()

    def mpso_ccd(self):
        """Runs PSO with CCD"""
        start = time.time()
        for _ in range(self.pso.mpso_runs):
            self.run_PSO()
            self.pso.g_best = ccd.CCD(
                initial=self.pso.g_best, lb = self.pso.lower_bound, ub = self.pso.upper_bound,
                alpha = self.pso.ccd_alpha, tol=self.pso.ccd_tol, max_its = self.pso.ccd_max_its,
                third_term_its=self.pso.ccd_third_term_its, func=self.pso.function)
            
        print(f"it took {time.time() - start} seconds to run")

    def mpso(self):
        """Runs MPSO without CCD"""
        start = time.time()
        for _ in range(self.pso.mpso_runs):
            self.run_PSO()
            
        print(f"it took {time.time() - start} seconds to run")


def test_PSO():

    pso = PSO(num_part = p.NUM_PART, num_dim=p.NUM_DIM, alpha = p.ALPHA, upper_bound=p.UPPER_BOUND, lower_bound=p.LOWER_BOUND,
        max_iterations=p.MAX_ITERATIONS, w=p.W, c1=p.C1, c2=p.C2, tolerance=p.TOLERANCE, mv_iteration=p.NO_MOVEMENT_TERMINATION,
        optimum=p.OPTIMUM, bias=p.BIAS, functionID=p.FUNCT, ccd_alpha=p.CCD_ALPHA, ccd_tol=p.CCD_TOL, ccd_max_its=p.CCD_MAX_ITS,
        ccd_third_term_its=p.CCD_THIRD_TERM_ITS, mpso_runs=30)
    
    import json

    #runner = PSORunner(pso)
    #runner.mpso_ccd()
    #print(runner.pso.g_best)

    

#test_PSO()





    #terminating conditions: will either terminate when
    #1. max iterations reached
    #2. minimal movements after a few iterations. If the g_best doesn't move much and fits within our tolerance, we
    # exit




