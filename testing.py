import update as up
import numpy as np

def TestVerifyBounds():
    array = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    lower = np.array((2, 2, 2))
    upper = np.array((6, 9, 8))
    array = up.verify_bounds(upper_bound = upper, lower_bound = lower, matrix = array)
    print(array)

TestVerifyBounds()