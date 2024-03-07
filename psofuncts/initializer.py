import numpy as np
import psofuncts.update as up
import parameters as p


def _default(array: np.ndarray[p.DTYPE]) -> p.DTYPE:
    """Private Function. Really Shouldn't be used.
    _Default is used as the default optimization function in initializer, this is really for debugging.
    Ideally, we will have other functions for more complex optimization problems that will take place of default"""

    #WOW the default function is used as a default input. I'm shocked!
    return 1



def initializer(num_part: int, num_dim: int, alpha: p.DTYPE, 
                upper_bound: p.ADTYPE, lower_bound: p.ADTYPE, 
                function = _default) -> (p.ADTYPE, p.ADTYPE, p.ADTYPE, p.ADTYPE, p.ADTYPE):
    """ Initialization function for the PSO algorithm. 

    ------------Parameters (input)------------
    
    [All parameter inputs for this function are stored in parameters.py]
    num_part:  Number of particles 
    num_dim: Number of dimensions
    alpha:  parameter for velocity max
    upper_bound:  Upper bounds of the domain of the problem
    lower_bound:  Lower bounds of the domain of the problem
    function: Optimization Problem (uses _Default function for no input)

    ------------Returns (Output)--------------

    pos_matrix: an ndarray that keep the position of each particle (Initialized with random values for each dimmension)
    vel_matrix: an ndarray that keeps the velocity for each particle (Initialized with random values for each dimmension)
    p_best: an ndarray that keeps the personal minimum for each particle in each dimmension
    g_best: an ndarray that keeps the global minimum between all particles in each dimmension
    v_max: float based on the size of the area, this is the max velocity each particle can move
    """

    # Randomly initialize the positions of each of the particles
    pos_matrix = _x_initializer(num_dim=num_dim, num_part=num_part, upper_bound=upper_bound, lower_bound=lower_bound)

    # Randomly assign velocities to each of the particles
    vel_matrix, v_max = _v_initializer(num_dim=num_dim, num_part=num_part, upper_bound=upper_bound, lower_bound=lower_bound, alpha = alpha)

    # The distances row contains the distances for each particle's p_best.  It is used to keep track of
    # Results so no recalculation is needed.  It is initialized at the max value, so that when the function
    # Is evaluated for the first time it properly updates
    evaluation_row = np.ones((1, num_part), dtype=p.DTYPE)
    evaluation_row *= np.finfo(p.DTYPE).max

    # Let the personal best be the current position.
    p_best = np.vstack((pos_matrix, evaluation_row), dtype=p.DTYPE)
    p_best = up.update_p_best(pos_matrix=pos_matrix, past_p_best=p_best, function=function)

    g_best = up.update_g_best(p_best=p_best)

    return pos_matrix, vel_matrix, p_best, g_best, v_max

def _x_initializer(num_dim: int, num_part: int, upper_bound: np.ndarray[p.DTYPE], lower_bound: np.ndarray[p.DTYPE]) -> np.ndarray:
    """Private function. Used in initializer. Randomly initializes the positions of each particle within the upper and lower bound limits of each dimmension"""
    scalingfactor = upper_bound - lower_bound

    pos_matrix = np.random.rand(num_dim, num_part)

    pos_matrix[ : ] *= scalingfactor[ : , np.newaxis]
    pos_matrix[ : ] += lower_bound[ : , np.newaxis]

    return pos_matrix

def _v_initializer(num_dim: int, num_part: int, upper_bound: p.ADTYPE, lower_bound: p.ADTYPE, alpha: p.DTYPE) -> (p.ADTYPE, p.ADTYPE):
    """Private function. Used in initializer. Randomly initializes the velocities of each particle
    """
    if alpha < 0 or alpha >= 1:
        raise Exception("Bad alpha parameter")
    
    v_max = alpha*(upper_bound - lower_bound)

    scalingfactor = 2*v_max

    vel_matrix = np.random.rand(num_dim, num_part)

    vel_matrix[ : ] *= scalingfactor[ : , np.newaxis]
    vel_matrix[ : ] -= (v_max)[ : , np.newaxis]

    return vel_matrix, v_max
