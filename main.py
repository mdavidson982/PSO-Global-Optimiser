import numpy as np
import initializer as ini
import parameters as p
import update as up
import testfuncts as tf
import time


def good_learning_parameters(w: p.DTYPE, c1: p.DTYPE, c2: p.DTYPE) -> bool:
    return w < 1 and w > 0.5*(c1+c2)

#initialization variables
#calls the initialization function in the initializer.py. 
#Uses the parameter values created in parameters.py
#Returns: pos_matrix, an ndarray that keep the position of each particle (Initialized with random values for each dimmension)
#         vel_matrix, an ndarray that keeps the velocity for each particle (Initialized with random values for each dimmension)
#         p_best, an ndarray that keeps the personal minimum for each particle in each dimmension
#         g_best, an ndarray that keeps the global minimum between all particles in each dimmension
#         v_max, float based on the size of the area, this is the max velocity each particle can move 


#Here we will place the MPSO function. To achieve this we will loop through the velocity update, position update,
#p_best, and g_best functions until we get a satisfactory answer, and loop under the number of times defined in
#max iteration var
def mpso(num_part: int, num_dim: int, alpha: p.DTYPE, 
                upper_bound: np.ndarray[p.DTYPE], lower_bound: np.ndarray[p.DTYPE], 
                max_iterations: int, w: p.DTYPE, c1: p.DTYPE, c2: p.DTYPE, tolerance: p.DTYPE,
                mv_iteration: int, function):
    start = time.time()
    
    if not good_learning_parameters(w, c1, c2):
        raise Exception("Bad parameters")
    
    # Run initialization to get necessary matrices
    pos_matrix, vel_matrix, p_best, g_best, v_max = ini.initializer(num_part=num_part, num_dim=num_dim, 
                                            alpha=alpha, upper_bound=upper_bound,
                                            lower_bound=lower_bound, function = function)

    # Store the image of past g_bests for the second terminating condition.  Multiplies a ones array by the maximum possible value
    # So that comparison starts out always being true.
    old_g_best = np.finfo(p.DTYPE).max*np.ones(mv_iteration, dtype=p.DTYPE)

    for i in range (max_iterations):
        vel_matrix = up.update_velocity(v_part=vel_matrix, x_pos=pos_matrix, g_best=g_best, p_best=p_best, w=w, c1=c1, c2=c2)
        vel_matrix = up.verify_bounds(upper_bound = v_max, lower_bound = -v_max, matrix = vel_matrix)
        pos_matrix = up.update_position(x_pos=pos_matrix, v_part=vel_matrix)
        #added verify bound to the MPSO loop. Assumed position matrix was the correct input. Putting this comment here to make sure that's right later when we review.
        pos_matrix = up.verify_bounds(upper_bound = upper_bound, lower_bound = lower_bound, matrix = pos_matrix)
        p_best = up.update_p_best(pos_matrix= pos_matrix, past_p_best = p_best, function = function)
        g_best = up.update_g_best(p_best=p_best)

        #if np.linalg.norm(g_best - old_g_best) < tolerance:
        #we could implement this by storing the last 15 vals in an array and then computing stdev

        """

        print("iteration: ", i+1 )
        print("This is the velocity matrix: ")
        print(vel_matrix)
        print("this is the position matrix: ")
        print(pos_matrix)
        print("this is the p best: ")
        print(p_best)
        print("this is the g best: ")
        print(g_best)

        """

        #roll simply shifts the numpy matrix over by 1.  So,
        # [1, 2, 3]
        # Where 1 might be the first image of g_best 2 might be the second image of g_best etc. becomes
        # [3, 1, 2]
        # We also want to store the image of the newest g_best.  This is done by replace what would be 3 in the above with the new image.
        # If 0 is the new image of g_best, then the above array becomes
        # [0, 1, 2] 
        old_g_best = np.roll(old_g_best, 1)
        old_g_best[0] = g_best[-1]

        #Second terminating condition.
        if (abs(old_g_best[0]-old_g_best[-1])/(abs(old_g_best[-1]) + tolerance)) < tolerance:
            break

        #input() #

        #Use function value instead of image
    print("The global best was ", g_best[:-1])
    print("The best value was ", g_best[-1])
    print(f"We had {i+1} iterations")
    print(f"The function took {time.time()-start} seconds to run")

mpso(num_part = p.NUM_PART, num_dim=p.NUM_DIM, alpha = p.ALPHA, upper_bound=p.UPPER_BOUND, lower_bound=p.LOWER_BOUND,
     max_iterations=p.MAX_ITERATIONS, w=p.W, c1=p.C1, c2=p.C2, tolerance=p.TOLERANCE, mv_iteration=p.NO_MOVEMENT_TERMINATION,
     function = tf.Sphere)



    #terminating conditions: will either terminate when
    #1. max iterations reached
    #2. minimal movements after a few iterations. If the g_best doesn't move much and fits within our tolerance, we
    # exit




