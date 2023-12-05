import numpy as np

DTYPE = np.float64 # Data type to be used.
ADTYPE = np.ndarray[DTYPE] #Alias of array datatype, for ease of use

#Number of dimensions of problem
NUM_DIM: int = 2
#Number of particles that we want to use
NUM_PART: int = 500

#Max number of iterations. This determines the number of times the MPSO algorithm updates
MAX_ITERATIONS: int = 1000

#Part of second termination criteria, how many indexes back from old_g_best we should check
NO_MOVEMENT_TERMINATION: int = 20

#Part of second termination criteria.  The tolerance which defines how small updates should be.
TOLERANCE: np.float64 = 10**-6

FUNCT = "sphere"

#Upper and lower bounds (U-L array).  Bound the domain of the function.
UPPER_BOUND = np.array([1, 1], dtype=DTYPE)
LOWER_BOUND = np.array([-1, -1], dtype=DTYPE)
OPTIMUM = np.array([0, 0], dtype=DTYPE)
BIAS: DTYPE = 45

#Velocity restrictor [0-1].  
ALPHA: np.float64 = 0.1

#Learning parameters:
#Weight part of inertia compoennet
W: np.float64 = 0.5 #velocity, Randomized from 0 - 1

#Acceleration coefficients
C1: np.float64 = 0.3 #Cognitive parameter.  Random number from [0 - 2), fixed throughout the function
C2: np.float64 = 0.3 #Social parameter.  Random number from [0 - 2), fixed throughout the function

