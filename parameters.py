import numpy as np

from dataclasses import dataclass

DTYPE = np.float64 # Data type to be used.
ADTYPE = np.ndarray[DTYPE] #Alias of array datatype, for ease of use

#Number of dimensions of problem
NUM_DIM: int = 5
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
UPPER_BOUND = np.ones(NUM_DIM, dtype=DTYPE)*100
LOWER_BOUND = UPPER_BOUND*-1
OPTIMUM = np.zeros(NUM_DIM, dtype=DTYPE)
BIAS: DTYPE = 0

MPSO_RUNS = 30

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




@dataclass        
class PSOHyperparameters:
    """Class which holds PSO hyperparameters

    num_part:           number of particles the instance will use
    num_dim:            dimensions of the problem to be solved
    alpha:              velocity constriction parameter
    max_iterations:     maximum number of iterations the instance can run through
    w:                  momentum parameter (see docs for more)
    c1:                 cognitive parameter (see docs for more)
    c2:                 social parameter (see docs for more)
    tolerance:          part of second termination criteria (see docs)
    mv_iteration:       part of second termination criteria (see docs)

    has_valid_learning_params():  Determines if the parameters are valid"""

    num_part: int
    num_dim: int
    alpha: p.ADTYPE
    w: p.DTYPE
    c1: p.DTYPE
    c2: p.DTYPE
    tolerance: p.DTYPE
    mv_iteration: int
    max_iterations: int

    def has_valid_learning_params(self) -> bool:
        if not (self.alpha < 1 and self.alpha > 0):
            return False
        if not (self.tolerance > 0):
            return False
        if not (self.num_part > 0):
            return False
        if not (self.num_dim > 0):
            return False
        if not (self.mv_iteration > 0):
            return False
        if not (self.max_iterations > 0):
            return False
        if not (self.c1 < 2 and self.c1 > 0):
            return False
        if not (self.c2 < 2 and self.c2 > 0):
            return False
        # See page 5 of manuscript for below condition
        if not (self.w < 1 and self.w > 0.5*(self.c1+self.c2)-1):
            return False
        return True
    
    def to_json(self):
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_data):
        dict = json.loads(json_data)
        return cls(**dict)



