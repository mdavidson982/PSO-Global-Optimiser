import numpy as np

def update_p_best(pos_matrix: np.ndarray, past_p_best: np.ndarray, function):
    results = np.apply_along_axis(function, axis=0, arr=pos_matrix)
    evaluated = np.vstack((pos_matrix, results))
    mask = past_p_best[-1, :] < evaluated[-1, :]
    return np.where(mask, past_p_best, evaluated)

def update_g_best(p_best: np.ndarray):
    # find minimum p_best
    # if minimum p_best is less than g_best, update that
    print(p_best[-1, :])
    return p_best[:, np.argmin(p_best[-1, :])]


def initializer(num_part: int, num_dim: int, alpha: np.float64, upper_bound: np.ndarray, lower_bound: np.ndarray):
    """ Initialization function for the PSO algorithm.
    num_part:  Number of particles
    num_dim: Number of dimensions
    alpha:  parameter for velocity max
    upper_bound:  Upper bounds of the domain of the problem
    lower_bound:  Lower bounds of the domain of the problem
    """

    # Randomly initialize the positions of each of the particles
    pos_matrix = _x_initializer(num_dim=num_dim, num_part=num_part, upper_bound=upper_bound, lower_bound=lower_bound)

    # Randomly assign velocities to each of the particles
    vel_matrix, v_max = _v_initializer(num_dim=num_dim, num_part=num_part, upper_bound=upper_bound, lower_bound=lower_bound, alpha = alpha)

    # The distances row contians the distances for each particle's p_best.  It is used to keep track of
    # Results so no recalculation is needed.
    distances_row = np.ones((1, num_part))
    distances_row *= np.finfo(np.float64).max

    # Let the personal best be the current position.
    p_best = np.vstack((pos_matrix, distances_row))

    #TODO:  make g_best its own function
    g_best = p_best[:, 0].copy()

    return pos_matrix, vel_matrix, p_best, g_best, v_max

def _x_initializer(num_dim: int, num_part: int, upper_bound: np.ndarray, lower_bound: np.ndarray):
    scalingfactor = upper_bound - lower_bound

    pos_matrix = np.random.rand(num_dim, num_part)

    pos_matrix[ : ] *= scalingfactor[ : , np.newaxis]
    pos_matrix[ : ] += lower_bound[ : , np.newaxis]

    return pos_matrix

def _v_initializer(num_dim: int, num_part: int, upper_bound: np.ndarray, lower_bound: np.ndarray, alpha: float):
    if alpha < 0 or alpha >= 1:
        raise Exception("Whomp whomp")
    
    v_max = alpha*(upper_bound - lower_bound)

    scalingfactor = 2*v_max

    vel_matrix = np.random.rand(num_dim, num_part)

    vel_matrix[ : ] *= scalingfactor[ : , np.newaxis]
    vel_matrix[ : ] -= (v_max)[ : , np.newaxis]

    return vel_matrix, v_max