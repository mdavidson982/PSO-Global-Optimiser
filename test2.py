import numpy as np
import time

start = time.time_ns()
z: np.float64 = 10**-6
print(time.time_ns() - start)

print(z)