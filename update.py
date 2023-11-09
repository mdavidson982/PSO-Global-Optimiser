import numpy as np
#Private Function. Really Shouldn't be used.
#This is used in place of the equation we will be optimizing.
#This will be changed out for some other hardcoded equation down the line. Right now, this is just a place holder
def _Default(array: np.ndarray) -> np.float64:
    return 1 


def update_p_best(pos_matrix: np.ndarray, past_p_best: np.ndarray, function = _Default) -> np.ndarray:
    """Updates the personal best for each particle in each iteration (if the updated value is smaller than the previous value).
    This is completed by preforming the optomization function on each particle

    Pos_matrix: ndarray that represents the current position of each particle
    Past_p_best: represents the previous personal best position for each particle
    Function: preforms optimization function, currently uses _Default as default input if not provided
    np.ndarray (return type): The following function returns an ndarray that repersents the updated personal best positions"""
    results = np.apply_along_axis(function, axis=0, arr=pos_matrix) # Run the function for every particle, and store the result in an array
    evaluated = np.vstack((pos_matrix, results)) # Append the results to the bottom of the position matrix
    mask = past_p_best[-1, :] < evaluated[-1, :] # Boolean mask for every row.  Basically, only update columns if the result is smaller
    return np.where(mask, past_p_best, evaluated) # Apply the mask

def update_g_best(p_best: np.ndarray) -> np.ndarray:
    """Updates the global minimum value found in the p_best for each particle.
    g_best is determined by selecting the particle with the best minimum in each dimmension"""
    # Since g_best should always be in p_best, return the min of p_best.

    #np.argmin(p_best[-1, :]) finds the index of the minmum value in the last row of p_best
        #Last row contains the objective function value for each particle (output of optimization function [I'm pretty sure])
    #p_best[:, np.argmin(p_best[-1, :])] selects the entire column corresponding to the index found earlier.
        #Selects the personal best positions of the particle with the best with the best minimum
    #.copy() ensures that the returned g_best ndarray is a seprate copy and not referenced to the original p_best ndarray
    return p_best[:, np.argmin(p_best[-1, :])].copy()


def update_velocity(v_part: np.ndarray, x_pos: np.ndarray, g_best: np.ndarray, 
                    p_best: np.ndarray, w: np.float64, c1: np.float64, c2: np.float64):
    """Updates the velocity for each particle in each dimmension
    v_part: ndarray that represents the current velocities for each dimmention of a particle
    x_pos: ndarray that reperesents the current position of the particle
    g_best: ndarray for the the global best position found by the particle swarm (seem more at update_g_best)
    p_best: ndarray for the personal best position found by each particle (Look at update p_best)
    w: ndarray representing inertia weight for the PSO algorithm
    c1: ndarray representing cognitive (personal) component [Stored in parameters.py]
    c2: ndarray representing social (global) component [Stored in paramtets.py]"""
    
    
    #Randomness variables. Returns values between 0 and 1.
    #Random movement influence. Technically not necessary but the random movement values
    #Make the movement for each particle more 'natural'
    r1 = np.random.rand()
    r2 = np.random.rand()

    print(f"r1 is: {r1}, r2 is {r2}")

    print("velocity was:")
    print(v_part)


    #Update Velocity Formula
    #v_part * w: Inertia term. It allows particles to retain some of their previous velocity
    #r1 * c1 * (x_pos - p_best[:-1]): Personal Cognitive Component. Pulls the particle towards the personal best (p_best) position
    #r2 * c2 * (x_pos - g_best[:-1, np.newaxis]): Global Cognitive Component. Pulls the partivle towards the global best position
    v_part = v_part*w + r1*c1*(x_pos-p_best[:-1]) + r2*c2*(x_pos-g_best[:-1, np.newaxis])

    print("velocity is now:")
    print(v_part)


def update_position(x_pos: np.ndarray, v_part: np.ndarray):
    """Updates the position of a particle by adding the velocity to the position for each dimmension
    returns an updated position ndarray"""
    return x_pos + v_part


def verify_bounds(upper_bound: np.ndarray, lower_bound: np.ndarray, matrix: np.ndarray):

    #The following function verifies that the matrix does not exceed the upper or lower bound dimmensions. 
    #Here's an example of constraining both Max and Min:
    """
        Original array:
        [[2 5 8]
        [1 4 6]
        [3 7 9]]
        Upper Bounds 1D Array:
        [3 6 7]
        Lower Bound 1D Array:
        [2 3 5]
        Result after applying minimum constraint:
        [[2 3 3]
        [3 4 6]
        [5 7 7]]"""
    
    #Pay attention to the the index of both the lower and upper bound 1D arrays. they coincide with the with the respective row in the 2D array
    #(Notice how row 3 in the original array changed from 3, 7, 9 to 5, 7, 7 
    #This is because 5 is the minimum value seen in the third index of the lower bound array and 7 is the max value in the upper bound array. 
    #Because of this, the 3 and 9 in the last row of the original array change to min and max respectively)
    result = np.maximum(matrix, lower_bound[: np.newaxis])
    result = np.minimum(matrix, upper_bound[:, np.newaxis])

    return result