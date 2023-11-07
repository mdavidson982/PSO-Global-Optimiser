import numpy as np
import initializer as ini
import time

def good_learning_parameters(w: float, c1: float, c2: float):
    return w < 1 and w > 0.5*(c1+c2)

num_dim = 4
num_part = 5

#Upper and lower bounds (U-L array) 1 for each dim
upper_bound = np.array([3, 7, 4, 8])
lower_bound = np.array([1, 2, 3, 4])

alpha: np.float64 = 0.3

#Learning parameters:
#Weight part of inertia compoennet
w: np.float64 = 0.7 #velocity, Randomized from 0 - 1

#Acceleration coefficients
c1: np.float64 = 0.3 #Random numbe from [0 - 2), fixed throughout the function,
c2: np.float64 = 0.4 #Random number from [0- 2), fixed throughout the function

if not good_learning_parameters(w, c1, c2):
    raise Exception("Bad parameters")

k = None #Step increment

#initialization variables
start = time.time_ns()
pos_matrix, vel_matrix, p_best, g_best, v_max = ini.initializer(num_part=num_part, num_dim=num_dim, alpha=alpha, upper_bound=upper_bound,
                                                                lower_bound=lower_bound)
print(f"initialization took {time.time_ns()-start} nanoseconds")

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

update_velocity(v_part=vel_matrix, x_pos=pos_matrix, g_best=g_best, p_best=p_best, w=w, c1=c1, c2=c2)