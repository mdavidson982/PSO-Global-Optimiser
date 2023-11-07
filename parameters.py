import numpy as np

#Number of dimensions of problem
NUM_DIM = 4
#Number of particles that we want to use
NUM_PART = 5

#Upper and lower bounds (U-L array).  Bound the domain of the function.
UPPER_BOUND = np.array([3, 7, 4, 8])
LOWER_BOUND = np.array([1, 2, 3, 4])

#Velocity restrictor [0-1].  
ALPHA: np.float64 = 0.3

#Learning parameters:
#Weight part of inertia compoennet
W: np.float64 = 0.7 #velocity, Randomized from 0 - 1

#Acceleration coefficients
C1: np.float64 = 0.3 #Cognitive parameter.  Random number from [0 - 2), fixed throughout the function
C2: np.float64 = 0.4 #Social parameter.  Random number from [0 - 2), fixed throughout the function