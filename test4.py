from psofuncts import ccd
import numpy as np

z = np.array((
    (1, 2, 3),
    (2, 3, 4)
))

z[:] *= np.array((1, 2))[:, np.newaxis]
print(z)