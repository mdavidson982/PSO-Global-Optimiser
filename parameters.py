import numpy as np

DTYPE = np.float64 # Data type to be used.
ADTYPE = np.ndarray[DTYPE] #Alias of array datatype, for ease of use

#Number of dimensions of problem
NUM_DIM: int = 30
#Number of particles that we want to use
NUM_PART: int = 50

#Max number of iterations. This determines the number of times the MPSO algorithm updates
MAX_ITERATIONS: int = 100

#Part of second termination criteria, how many indexes back from old_g_best we should check
NO_MOVEMENT_TERMINATION: int = 20

#Part of second termination criteria.  The tolerance which defines how small updates should be.
TOLERANCE: np.float64 = 10**-6

FUNCT = "rosenbrock"

#Upper and lower bounds (U-L array).  Bound the domain of the function.
UPPER_BOUND = np.ones(NUM_DIM, dtype=DTYPE)*1000
LOWER_BOUND = UPPER_BOUND*-1
OPTIMUM = np.zeros(NUM_DIM, dtype=DTYPE)
BIAS: DTYPE = 45

#Velocity restrictor [0-1].  
ALPHA: np.float64 = 0.9

#Learning parameters:
#Weight part of inertia compoennet
W: np.float64 = 0.5 #velocity, Randomized from 0 - 1

#Acceleration coefficients
C1: np.float64 = 0.4 #Cognitive parameter.  Random number from [0 - 2), fixed throughout the function
C2: np.float64 = 0.4 #Social parameter.  Random number from [0 - 2), fixed throughout the function

###########################################
#CCD parameters
CCD_ALPHA = 0.2
CCD_MAX_ITS = 20
CCD_TOL: DTYPE = 10**-6
CCD_THIRD_TERM_ITS = 5
