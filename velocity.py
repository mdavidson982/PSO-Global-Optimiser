import numpy as np
import random

num_dim = 4
num_part = 5

#Upper and lower bounds (U-L array) 1 for each dim
upper_bound = np.array([3, 7, 4, 8])
lower_bound = np.array([1, 2, 3, 4])

alpha = 0.3

def x_initializer():
    #in a real problem, we'll have to define each scaling factor before we calculate ?
    scalingfactor = upper_bound - lower_bound

    pos_matrix = np.random.rand(num_dim, num_part)

    pos_matrix[ : ] *= scalingfactor[ : , np.newaxis]
    pos_matrix[ : ] += lower_bound[ : , np.newaxis]

    return pos_matrix

def v_initializer(alpha: float):
    if alpha < 0 or alpha >= 1:
        raise Exception("Whomp whomp")
    
    vmax = upper_bound - lower_bound

    scalingfactor = 2*vmax

    vel_matrix = np.random.rand(num_dim, num_part)

    vel_matrix[ : ] *= scalingfactor[ : , np.newaxis]
    vel_matrix[ : ] += alpha*(-1*vmax)[ : , np.newaxis]

    return vel_matrix, vmax

def initializer():
    pos_matrix = x_initializer()
    vel_matrix, vmax = v_initializer(alpha = alpha)
    pbest = x_initializer()
    gbest = np.random.rand(num_dim, 1)

initializer()


    


def good_learning_parameters(w: float, c1: float, c2: float):
    if w < 1 and w > 0.5*(c1+c2):
        return True
    return False

if not good_learning_parameters(w, c1, c2):
    raise Exception("Bad parameters")

#Learning parameters:
#Weight part of inertia compoennet
w = 0.7 #velocity, Randomized from 0 - 1

#Acceleration coefficients
c1 = 0.3 #Random numbe from [0 - 2), fixed throughout the function,
c2 = 0.4 #Random number from [0- 2), fixed throughout the function

k = None #Step increment

#Randomize velocity vector
g_best = np.random.rand(num_dim, 1)
p_best = initializer()
x_pos = initializer()
v_part = initializer()

def update_velocity(v_part, x_pos, g_best, p_best, w, c1, c2):
#Randomness variables
    r1 = random.random()
    r2 = random.random()

    print(f"r1 is: {r1}, r2 is {r2}")

    print("velocity was:")
    print(v_part)

    for i in range(num_part):
        # The three parts below are, Inertia, cognitive path, social/global part
        v_part[:, i] = (w*v_part[:, i]) + r1*c1*(x_pos[:, i] - p_best[:, i]) + r2*c2*(x_pos[:, i] - g_best[:, 0])

    print("velocity is now:")
    print(v_part)

update_velocity(v_part=v_part, x_pos=x_pos, g_best=g_best, p_best=p_best, w=w, c1=c1, c2=c2)