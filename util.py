import parameters as p
import numpy as np


def Scale(lb: p.ADTYPE, ub: p.ADTYPE, array) -> p.ADTYPE:
    """
    Take an array of values 0-1, and project them to the new space.
    """
    scale_factor = ub-lb
    array[:] *= scale_factor[:, np.newaxis]
    array[:] += lb[:, np.newaxis]
    return array

def Descale(lb: p.ADTYPE, ub: p.ADTYPE, array) -> p.ADTYPE:
    scale_factor = ub-lb
    array[:] -= lb[:, np.newaxis]
    array[:] /= scale_factor[:, np.newaxis]
    return array

def Project(old_lb: p.ADTYPE, old_ub: p.ADTYPE, new_lb: p.ADTYPE, new_ub: p.ADTYPE, array) -> p.ADTYPE:
    array = Descale(lb = old_lb, ub = old_ub, array=array)
    return Scale(lb=new_lb, ub=new_ub, array=array)

def TestScale():
    test_lb = np.array((-1, -3), dtype = p.DTYPE)
    test_ub = np.array((5, 8), dtype=p.DTYPE)
    array = np.random.rand(2, 5)

def TestDescale():
    test_lb = np.array((-1, -3), dtype = p.DTYPE)
    test_ub = np.array((5, 8), dtype=p.DTYPE)
    array=np.array(((4.29238699,  4.94183464,  4.01325885,  4.43113782,  4.09668852),
                    (-2.16691134,  5.27488954, 5.05971429,  4.89988192,  2.01928249)), dtype=p.DTYPE)

def TestProject():
    test_old_lb = np.array((-5, -4))
    test_old_ub = np.array((5, 4))

    test_new_lb = np.array((10, 8))
    test_new_ub = np.array((20, 16))

    array = np.random.rand(2, 5)
    array = Scale(test_old_lb, test_old_ub, array)

    print(array)
    print(Project(test_old_lb, test_old_ub, test_new_lb, test_new_ub, array))
