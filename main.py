import numpy as np
import initializer as ini
import parameters as p
import update as up



def good_learning_parameters(w: float, c1: float, c2: float):
    return w < 1 and w > 0.5*(c1+c2)



if not good_learning_parameters(p.W, p.C1, p.C2):
    raise Exception("Bad parameters")

#initialization variables
pos_matrix, vel_matrix, p_best, g_best, v_max = ini.initializer(num_part=p.NUM_PART, num_dim=p.NUM_DIM, alpha=p.ALPHA, upper_bound=p.UPPER_BOUND,
                                                                lower_bound=p.LOWER_BOUND)


up.update_velocity(v_part=vel_matrix, x_pos=pos_matrix, g_best=g_best, p_best=p_best, w=p.W, c1=p.C1, c2=p.C2)

