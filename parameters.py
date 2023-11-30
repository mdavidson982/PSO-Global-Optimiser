import numpy as np

#Number of dimensions of problem
NUM_DIM: int = 2
#Number of particles that we want to use
NUM_PART: int = 5

#Max number of iterations. This determines the number of times the MPSO algorithm updates
MAX_ITERATIONS: int = 1000

#Part of second termination criteria, how many indexes back from old_g_best we should check
NO_MOVEMENT_TERMINATION: int = 1000

#Part of second termination criteria.  The tolerance which defines how small updates should be.
TOLERANCE: np.float64 = 10**-6

#Upper and lower bounds (U-L array).  Bound the domain of the function.
UPPER_BOUND = np.array([100, 100], dtype=np.float64)
LOWER_BOUND = np.array([-100, -100], dtype=np.float64)

#Velocity restrictor [0-1].  
ALPHA: np.float64 = 0.1

#Learning parameters:
#Weight part of inertia compoennet
W: np.float64 = 0.5 #velocity, Randomized from 0 - 1

#Acceleration coefficients
C1: np.float64 = 0.1 #Cognitive parameter.  Random number from [0 - 2), fixed throughout the function
C2: np.float64 = 0.1 #Social parameter.  Random number from [0 - 2), fixed throughout the function

DTYPE = np.float64
ADTYPE = np.ndarray[DTYPE]