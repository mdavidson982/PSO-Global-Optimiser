from dataclasses import dataclass
import utils.parameters as p
import numpy as np
import time
import enum
from typing import get_origin

# This array will get filled in by the PsoDataClass from below
DATACLASSES = []

def PsoDataClass(cls):
    """Decorator used to add information to the decoding registry for a given class."""
    decode_json_hooks = {}
    """Detect if any special decoding needs to take place"""
    for key, value in cls.__annotations__.items():
        # Case 1: np arrays get converted to lists during jsonization.  If a field exists in a class
        # That is an np array, make sure that it is converted back to a numpy array.
        if get_origin(value) == np.ndarray: 
            decode_json_hooks[key] = np.array

    setattr(cls, "decode_json_hooks", decode_json_hooks)
    DATACLASSES.append(cls) # Add this specific 
    return cls

@PsoDataClass
@dataclass
class PSOConfig:
    """Class which holds any configurations for PSO besides hyperparameters, domain data etc.
    seed:               seed which is used to generate random numbers.
    """
    seed: int = int(time.time())
    
@PsoDataClass
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

    has_valid_learning_params():  Determines if the parameters are valid
    """

    num_part: int
    num_dim: int
    alpha: p.DTYPE
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
    
@PsoDataClass
@dataclass    
class CCDHyperparameters:
    """
    ccd_alpha:          Restricts the domain for ccd
    ccd_max_its:        Number of max iterations for ccd
    ccd_tol:            Tolerance parameter to determine if ccd is improving or not improving
    ccd_third_term_its: Number of iterations to determine if tolerance threshold reached
    
    has_valid_learning_parms():  Determines if the parameters are valid
    """

    ccd_alpha: p.DTYPE
    ccd_max_its: int
    ccd_tol: p.DTYPE
    ccd_third_term_its: int

    def has_valid_learning_params(self) -> bool:
        if not (self.ccd_alpha < 1 and self.ccd_alpha > 0):
            return False
        if not (self.ccd_max_its > 0):
            return False
        if not (self.ccd_tol > 0):
            return False
        if not (self.ccd_third_term_its > 0):
            return False
        return True

@PsoDataClass 
@dataclass
class DomainData:
    """
    Class that holds information about the objective function's
    domain for PSO

    upper_bound:        upper bounds of the domain
    lower_bound:        lower bounds of the domain
    v_max:              vector controlling the maximum velocity of particles
    """
    upper_bound: p.ADTYPE
    lower_bound: p.ADTYPE

@PsoDataClass
@dataclass
class IterationData:
    """
    Class that holds information about a specific iteration of PSO

    iteration_num:      iteration of pso
    pos_matrix:         matrix recording the position of the particles
    vel_matrix:         matrix recording the velocity of the particles
    p_best:             matrix recording the personal best of the particles
    recorded_g_best:    vector of the g_best that was stored in PSO.  Because of how
                        MPSO works, it may be that the g_best that an iteration found is not
                        better than a previous run of PSO.
    iteration_g_best:   Opposite of recorded_g_best, records the g_best that this particular
                        instance found before comparison with other iterations.

    old_g_bests:        All of the old g_bests that were recorded up to this point.
    """
    iteration_num: int
    pos_matrix: p.ADTYPE
    vel_matrix: p.ADTYPE
    p_best: p.ADTYPE
    recorded_g_best: p.ADTYPE
    iteration_g_best: p.ADTYPE
    old_g_bests: p.ADTYPE

class PSOLogTypes(enum.Enum):
    NO_LOG = 0
    QUALITY = 1
    TIME = 2
    
@PsoDataClass
@dataclass
class PSOLoggerConfig:
    """
    Settings for the PSOLogger class.
    """
    log_type: PSOLogTypes = PSOLogTypes.NO_LOG
    track_pos_matrix: bool = True
    track_vel_matrix: bool = True
    track_pbest_matrix: bool = True
    notes: str = ""

@PsoDataClass
@dataclass
class MPSORunnerConfigs:
    """Class which defines the configurations for the MPSO Runner"""
    use_ccd: bool = True

