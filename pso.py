import numpy as np
import initializer as ini
import parameters as p
import update as up
import testfuncts as tf
import time

def validate_learn_params(w: p.DTYPE, c1: p.DTYPE, c2: p.DTYPE) -> bool:
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
    functionID: int | str

    def __init__(self, num_part: int, num_dim: int, alpha: p.DTYPE, 
                upper_bound: p.ADTYPE, lower_bound: p.ADTYPE, 
                max_iterations: int, w: p.DTYPE, c1: p.DTYPE, c2: p.DTYPE, tolerance: p.DTYPE,
                mv_iteration: int, optimum: p.ADTYPE, bias: p.DTYPE, functionID: str | int):
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
        self.function = tf.TF.generate_function(functionID=functionID, optimum=optimum, bias=bias)

        if not validate_learn_params(w, c1, c2):
            raise Exception("Bad learning parameters")

    def initialize(self):
        # Run initialization to get necessary matrices
        self.pos_matrix, self.vel_matrix, self.p_best, self.g_best, self.v_max = ini.initializer(
            num_part=self.num_part, num_dim=self.num_dim, alpha=self.alpha, upper_bound=self.upper_bound,
            lower_bound=self.lower_bound, function = self.function)
        # Store the image of past g_bests for the second terminating condition.  Multiplies a ones array by the maximum possible value
        # So that comparison starts out always being true.
        self.old_g_best = np.finfo(p.DTYPE).max*np.ones(self.mv_iteration, dtype=p.DTYPE)
        self.iterations = 0

    def update(self) -> bool:
        self.vel_matrix = up.update_velocity(self.vel_matrix, self.pos_matrix, self.g_best, self.p_best, self.w, self.c1, self.c2)
        self.vel_matrix = up.verify_bounds(self.v_max, -self.v_max, self.vel_matrix)
        self.pos_matrix = up.update_position(self.pos_matrix, self.vel_matrix)
        #added verify bound to the pso loop. Assumed position matrix was the correct input. Putting this comment here to make sure that's right later when we review.
        self.pos_matrix = up.verify_bounds(upper_bound = self.upper_bound, lower_bound = self.lower_bound, matrix = self.pos_matrix)
        self.p_best = up.update_p_best(pos_matrix= self.pos_matrix, past_p_best = self.p_best, function = self.function)
        self.g_best = up.update_g_best(p_best=self.p_best)

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
        return self.iterations >= self.max_iterations or self.second_termination()
    
    def second_termination(self) -> bool:
        return (abs(self.old_g_best[0]-self.old_g_best[-1])/(abs(self.old_g_best[-1]) + self.tolerance)) < self.tolerance

# Non-graphical runner
class PSORunner:
    pso: PSO

    def __init__(self, pso: PSO):
        self.pso = pso

    # Run PSO manually.
    def run_PSO(self):
        start = time.time()
        self.pso.initialize()
        shouldTerminate = False
        while not shouldTerminate:
            shouldTerminate = self.pso.update()

        print("The global best was ", self.pso.g_best[:-1])
        print("The best value was ", self.pso.g_best[-1])
        print(f"We had {self.pso.iterations} iterations")
        print(f"The function took {time.time()-start} seconds to run")



        



"""
pso(num_part = p.NUM_PART, num_dim=p.NUM_DIM, alpha = p.ALPHA, upper_bound=p.UPPER_BOUND, lower_bound=p.LOWER_BOUND,
     max_iterations=p.MAX_ITERATIONS, w=p.W, c1=p.C1, c2=p.C2, tolerance=p.TOLERANCE, mv_iteration=p.NO_MOVEMENT_TERMINATION,
     enable_visualizer=True, function = tf.Sphere)
"""

def test_PSO():


    pso = PSO(num_part = p.NUM_PART, num_dim=p.NUM_DIM, alpha = p.ALPHA, upper_bound=p.UPPER_BOUND, lower_bound=p.LOWER_BOUND,
        max_iterations=p.MAX_ITERATIONS, w=p.W, c1=p.C1, c2=p.C2, tolerance=p.TOLERANCE, mv_iteration=p.NO_MOVEMENT_TERMINATION,
        optimum=p.OPTIMUM, bias=p.BIAS, functionID=p.FUNCT)

    runner = PSORunner(pso)
    runner.run_PSO()

test_PSO()





    #terminating conditions: will either terminate when
    #1. max iterations reached
    #2. minimal movements after a few iterations. If the g_best doesn't move much and fits within our tolerance, we
    # exit




