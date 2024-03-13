"various utility functions that have no other home for PSO"

import utils.parameters as p
import numpy as np
import utils.consts as c
import os
import time
import json

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
    
def np_to_json(arr: np.ndarray):
    return json.dumps(arr.tolist())

def np_from_json(s: str):
    return np.array(json.loads(s))

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

