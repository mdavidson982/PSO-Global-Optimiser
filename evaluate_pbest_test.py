import numpy as np
import initializer as ini

def TestFunction(arr: np.ndarray):
    return arr.sum()

num_part = 5
num_dim = 4
alpha = 0.3
upper_bound = np.array([3, 7, 4, 8])
lower_bound = np.array([1, 2, 3, 4])

pos_matrix, vel_matrix, p_best, g_best, v_max = ini.initializer(num_part=num_part, num_dim=num_dim, alpha=alpha, upper_bound=upper_bound,
                                                                lower_bound=lower_bound)

p_best = ini.update_p_best(pos_matrix=pos_matrix, past_p_best=p_best, function = TestFunction)
g_best = ini.update_g_best(p_best=p_best)

print(g_best)