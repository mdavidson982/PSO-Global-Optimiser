import parameters as p
import numpy as np

def scale(lb: p.ADTYPE, ub: p.ADTYPE, array: p.ADTYPE) -> p.ADTYPE:
    """
    Take an array of values 0-1, and project them to the new space.
    """
    scale_factor = ub-lb
    new_array = array.copy()
    new_array[:] *= scale_factor[:, np.newaxis]
    new_array[:] += lb[:, np.newaxis]
    return new_array

def descale(lb: p.ADTYPE, ub: p.ADTYPE, array: p.ADTYPE) -> p.ADTYPE:
    scale_factor = ub-lb
    new_array = array.copy()
    new_array -=  lb[:, np.newaxis]
    new_array[:] /= scale_factor[:, np.newaxis]
    return new_array

def project(old_lb: p.ADTYPE, old_ub: p.ADTYPE, new_lb: p.ADTYPE, new_ub: p.ADTYPE, array: p.ADTYPE) -> p.ADTYPE:
    array = descale(lb = old_lb, ub = old_ub, array=array)
    return scale(lb=new_lb, ub=new_ub, array=array)

def check_dim(array: p.ADTYPE, dim: int) -> bool:
    return array.shape[0] == dim

def dimension_to_xy_bounds(lb: p.ADTYPE, ub: p.ADTYPE) -> (p.ADTYPE, p.ADTYPE):
    """Function that takes a lower bound and upper bound vector, and converts them to the x and y upper bounds.\n
    E.g. if lb = [0, 1] and ub = [9, 7] then the return vectors would be \n
    x = [0, 9], y = [1, 7]
    """
    if not check_dim(lb, 2) or not check_dim(ub, 2):
        raise Exception("Improper dimensions")
    return np.array(lb[0], ub[0]), np.array[lb[1], ub[1]]

def test_scale():
    test_lb = np.array((-1, -3), dtype = p.DTYPE)
    test_ub = np.array((5, 8), dtype=p.DTYPE)
    array = np.random.rand(2, 5)

def test_descale():
    test_lb = np.array((-1, -3), dtype = p.DTYPE)
    test_ub = np.array((5, 8), dtype=p.DTYPE)
    array=np.array(((4.29238699,  4.94183464,  4.01325885,  4.43113782,  4.09668852),
                    (-2.16691134,  5.27488954, 5.05971429,  4.89988192,  2.01928249)), dtype=p.DTYPE)

def test_project():
    test_old_lb = np.array((-5, -4))
    test_old_ub = np.array((5, 4))

    test_new_lb = np.array((10, 8))
    test_new_ub = np.array((20, 16))

    array = np.random.rand(2, 5)
    array = scale(test_old_lb, test_old_ub, array)

    print(array)
    print(project(test_old_lb, test_old_ub, test_new_lb, test_new_ub, array))
