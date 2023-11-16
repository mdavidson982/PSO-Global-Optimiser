import numpy as np
import parameters as p

#putting a test function here
def Spherefunct(array: np.ndarray[p.DTYPE]) -> np.float64:
    return np.sum(array ** 2)