"various utility functions that have no other home for PSO"

import parameters as p
import numpy as np
import consts as c
import os
import time

def scale(lb: p.ADTYPE, ub: p.ADTYPE, array: p.ADTYPE) -> p.ADTYPE:
    """
    Take an array of values 0-1, and scale them to a new space. 
    Works in tandem with descale and project functions.
    """
    scale_factor = ub-lb
    if array.ndim == 2:
        new_array = array.copy()
        new_array[:] *= scale_factor[:, np.newaxis]
        new_array[:] += lb[:, np.newaxis]
        return new_array
    if array.ndim == 1:
         return array*scale_factor + lb

def descale(lb: p.ADTYPE, ub: p.ADTYPE, array: p.ADTYPE) -> p.ADTYPE:
    """
    Take an array of values, and descale them to be 0-1.
    Works in tandem with scale and project functions.
    """
    scale_factor = ub-lb
    if array.ndim == 2:
        new_array = array.copy()
        new_array -=  lb[:, np.newaxis]
        new_array[:] /= scale_factor[:, np.newaxis]
        return new_array
    if array.ndim == 1:
        return (array-lb)/scale_factor

def project(old_lb: p.ADTYPE, old_ub: p.ADTYPE, new_lb: p.ADTYPE, new_ub: p.ADTYPE, array: p.ADTYPE) -> p.ADTYPE:
    """Project an array of values from one domain to a new one."""
    array = descale(lb = old_lb, ub = old_ub, array=array)
    return scale(lb=new_lb, ub=new_ub, array=array)

def check_dim(array: p.ADTYPE, dim: int) -> bool:
    """Determine if the dimensions of the array are equal to the test dimensions"""
    return array.shape[0] == dim

def dimension_to_xy_bounds(lb: p.ADTYPE, ub: p.ADTYPE) -> (p.ADTYPE, p.ADTYPE):
    """Function that takes a lower bound and upper bound vector, and converts them to the x and y upper bounds.\n
    E.g. if lb = [0, 1] and ub = [9, 7] then the return vectors would be \n
    x = [0, 9], y = [1, 7]
    """
    if not check_dim(lb, 2) or not check_dim(ub, 2):
        raise Exception("Improper dimensions")
    
    return np.array((lb[c.XDIM], ub[c.XDIM])), np.array((lb[c.YDIM], ub[c.YDIM]))

def clear_temp() -> None:
    """Clears the temp directory of png files used in the visualizer"""
    files = os.listdir(c.TEMP_PATH)

    for file_name in files:
        file_path = os.path.join(c.TEMP_PATH, file_name)
        try:
            os.remove(file_path)
        except Exception:
            print(f"could not delete file {file_name}")

def make_tempfile_path() -> str:
    """Canonical temp path for png files"""
    return os.path.join(c.TEMP_PATH, f"TEMP{time.time_ns()}")

"""Below are functions used to test projections"""

def test_scale():
    
    test_lb = np.array((-1, -3), dtype = p.DTYPE)
    test_ub = np.array((5, 8), dtype=p.DTYPE)
    array = np.random.rand(2, 5)
    print(scale(test_lb, test_ub, array))

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

def test_scale_singledim():
    array = np.random.rand(2)
    test_lb = np.array((-1, -3), dtype = p.DTYPE)
    test_ub = np.array((5, 8), dtype=p.DTYPE)

    print(array)
    print(scale(test_lb, test_ub, array))

def test_descale():
    test_lb = np.array((-1, -3), dtype = p.DTYPE)
    test_ub = np.array((5, 8), dtype=p.DTYPE)
    array=np.array((-0.5, 7))
    print(descale(test_lb, test_ub, array))