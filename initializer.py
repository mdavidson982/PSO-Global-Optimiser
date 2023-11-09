import numpy as np
import update as up


def _Default(array: np.ndarray) -> np.float64:
    """Private Function. Really Shouldn't be used.
    _Default is used as the default optimization function in initializer, this is really for debugging.
    Ideally, we will have other functions for more complex optimization problems that will take place of default"""

    #WOW the default function is used as a default input. I'm shocked!
    return 1



def initializer(num_part: int, num_dim: int, alpha: np.float64, 
                upper_bound: np.ndarray, lower_bound: np.ndarray, 
                function = _Default) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """ Initialization function for the PSO algorithm. 

    ------------Parameters (input)------------
    
    [All parameter inputs for this function are stored in parameters.py]
    num_part:  Number of particles 
    num_dim: Number of dimensions
    alpha:  parameter for velocity max
    upper_bound:  Upper bounds of the domain of the problem
    lower_bound:  Lower bounds of the domain of the problem

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

    # The distances row contians the distances for each particle's p_best.  It is used to keep track of
    # Results so no recalculation is needed.  It is initialized at the max value, so that when the function
    # Is evaluated for the first time it properly updates
    distances_row = np.ones((1, num_part))
    distances_row *= np.finfo(np.float64).max

    # Let the personal best be the current position.
    p_best = np.vstack((pos_matrix, distances_row))
    p_best = up.update_p_best(pos_matrix=pos_matrix, past_p_best=p_best, function=function)

    g_best = up.update_g_best(p_best=p_best)

    return pos_matrix, vel_matrix, p_best, g_best, v_max

def _x_initializer(num_dim: int, num_part: int, upper_bound: np.ndarray, lower_bound: np.ndarray) -> np.ndarray:
    """Private function. Used in initializer. Randomly initializes the positions of each particle within the upper and lower bound limits of each dimmension"""
    scalingfactor = upper_bound - lower_bound

    pos_matrix = np.random.rand(num_dim, num_part)

    pos_matrix[ : ] *= scalingfactor[ : , np.newaxis]
    pos_matrix[ : ] += lower_bound[ : , np.newaxis]

    return pos_matrix

def _v_initializer(num_dim: int, num_part: int, upper_bound: np.ndarray, lower_bound: np.ndarray, alpha: np.float64) -> (np.ndarray, np.ndarray):
    """Private function. Used in initializer. Randomly initializes the velocities of each particle
    """
    if alpha < 0 or alpha >= 1:
        raise Exception("Whomp whomp")
    
    v_max = alpha*(upper_bound - lower_bound)

    scalingfactor = 2*v_max

    vel_matrix = np.random.rand(num_dim, num_part)

    vel_matrix[ : ] *= scalingfactor[ : , np.newaxis]
    vel_matrix[ : ] -= (v_max)[ : , np.newaxis]

    return vel_matrix, v_max