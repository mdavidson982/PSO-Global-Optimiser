import numpy as np


def update_p_best(pos_matrix: np.ndarray, past_p_best: np.ndarray, function) -> np.ndarray:
    results = np.apply_along_axis(function, axis=0, arr=pos_matrix) # Run the function for every particle, and store the result in an array
    evaluated = np.vstack((pos_matrix, results)) # Append the results to the bottom of the position matrix
    mask = past_p_best[-1, :] < evaluated[-1, :] # Boolean mask for every row.  Basically, only update columns if the result is smaller
    return np.where(mask, past_p_best, evaluated) # Apply the mask

def update_g_best(p_best: np.ndarray) -> np.ndarray:
    # Since g_best should always be in p_best, return the min of p_best.
    return p_best[:, np.argmin(p_best[-1, :])].copy()
