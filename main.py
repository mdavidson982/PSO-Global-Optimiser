import numpy as np
import initializer as ini
import parameters as p
import update as up



def good_learning_parameters(w: float, c1: float, c2: float):
    return w < 1 and w > 0.5*(c1+c2)



if not good_learning_parameters(p.W, p.C1, p.C2):
    raise Exception("Bad parameters")

#initialization variables
#calls the initialization function in the initializer.py. 
#Returns: pos_matrix, an ndarray that keep the position of each particle (Initialized with random values for each dimmension)
#         vel_matrix, an ndarray that keeps the velocity for each particle (Initialized with random values for each dimmension)
#         p_best, an ndarray that keeps the personal minimum for each particle in each dimmension
#         g_best, an ndarray that keeps the global minimum between all particles in each dimmension
#         v_max, float based on the size of the area, this is the max velocity each particle can move 
pos_matrix, vel_matrix, p_best, g_best, v_max = ini.initializer(num_part=p.NUM_PART, num_dim=p.NUM_DIM, alpha=p.ALPHA, upper_bound=p.UPPER_BOUND,
                                                                lower_bound=p.LOWER_BOUND)


up.update_velocity(v_part=vel_matrix, x_pos=pos_matrix, g_best=g_best, p_best=p_best, w=p.W, c1=p.C1, c2=p.C2)


#Here we will place the MPSO function. To achieve this we will loop through the velocity update, position update,
#p_best, and g_best functions until we get a satisfactory answer, and loop under the number of times defined in
#max iteration var

for p.MAX_ITERATIONS:
