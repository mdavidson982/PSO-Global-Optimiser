import numpy as np
import initializer as ini
import parameters as p

def good_learning_parameters(w: float, c1: float, c2: float):
    return w < 1 and w > 0.5*(c1+c2)



if not good_learning_parameters(p.W, p.C1, p.C2):
    raise Exception("Bad parameters")

#initialization variables
pos_matrix, vel_matrix, p_best, g_best, v_max = ini.initializer(num_part=p.NUM_PART, num_dim=p.NUM_DIM, alpha=p.ALPHA, upper_bound=p.UPPER_BOUND,
                                                                lower_bound=p.LOWER_BOUND)

def update_velocity(v_part: np.ndarray, x_pos: np.ndarray, g_best: np.ndarray, 
                    p_best: np.ndarray, w: np.float64, c1: np.float64, c2: np.float64):
#Randomness variables
    r1 = np.random.rand()
    r2 = np.random.rand()

    print(f"r1 is: {r1}, r2 is {r2}")

    print("velocity was:")
    print(v_part)
    v_part = v_part*w + r1*c1*(x_pos-p_best[:-1]) + r2*c2*(x_pos-g_best[:-1, np.newaxis])

    print("velocity is now:")
    print(v_part)

update_velocity(v_part=vel_matrix, x_pos=pos_matrix, g_best=g_best, p_best=p_best, w=p.W, c1=p.C1, c2=p.C2)