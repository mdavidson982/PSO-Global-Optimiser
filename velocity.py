import numpy as np
import random
import initializer as ini

def update_p_best(pos_matrix: np.ndarray, p_best: np.ndarray, function):
    pass

def good_learning_parameters(w: float, c1: float, c2: float):
    return w < 1 and w > 0.5*(c1+c2)

num_dim = 4
num_part = 5

#Upper and lower bounds (U-L array) 1 for each dim
upper_bound = np.array([3, 7, 4, 8])
lower_bound = np.array([1, 2, 3, 4])

alpha = 0.3

#Learning parameters:
#Weight part of inertia compoennet
w = 0.7 #velocity, Randomized from 0 - 1

#Acceleration coefficients
c1 = 0.3 #Random numbe from [0 - 2), fixed throughout the function,
c2 = 0.4 #Random number from [0- 2), fixed throughout the function

if not good_learning_parameters(w, c1, c2):
    raise Exception("Bad parameters")

k = None #Step increment

#Randomize velocity vector

pos_matrix, vel_matrix, p_best, g_best, v_max = ini.initializer(num_part=num_part, num_dim=num_dim, alpha=alpha, upper_bound=upper_bound,
                                                                lower_bound=lower_bound)

def update_velocity(v_part, x_pos, g_best, p_best, w, c1, c2):
#Randomness variables
    r1 = random.random()
    r2 = random.random()

    print(f"r1 is: {r1}, r2 is {r2}")

    print("velocity was:")
    print(v_part)

    for i in range(num_part):
        # The three parts below are, Inertia, cognitive path, social/global part
        v_part[:, i] = (w*v_part[:, i]) + r1*c1*(x_pos[:, i] - p_best[:-1, i]) + r2*c2*(x_pos[:, i] - g_best[:, 0])

    print("velocity is now:")
    print(v_part)

update_velocity(v_part=vel_matrix, x_pos=pos_matrix, g_best=g_best, p_best=p_best, w=w, c1=c1, c2=c2)