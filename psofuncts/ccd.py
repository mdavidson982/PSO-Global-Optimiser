import numpy as np
from scipy import optimize
import utils.parameters as p
import time

def CCD(initial: p.DTYPE, lb: p.ADTYPE, ub: p.ADTYPE, 
        alpha: p.DTYPE, tol: p.DTYPE, max_its: int, third_term_its: int, func):
    omega = alpha*(ub-lb) # Set restricted domain, as per step 2 on page 11 of manual

    q = initial.copy()[:-1] #Copy so that initial is not overwritten, and throw away the last value

    #Used to keep track of old f(q) coordinates for termination 3 criteria
    old_bests = np.finfo(p.DTYPE).max*np.ones(third_term_its, dtype=p.DTYPE)

    # Set I low and I high, as per step 3 on page 11 of manual
    I_low: np.ndarray = np.max(np.vstack((lb[np.newaxis], q[np.newaxis] - omega)), axis=0)
    I_high: np.ndarray = np.min(np.vstack((ub[np.newaxis], q[np.newaxis] + omega)), axis=0)
    I = np.vstack((I_low[np.newaxis], I_high[np.newaxis]))
    
    # l1 indicates the sequence of coordinates to perform brent on.  As per Step F, 
    # CCD should be performed forwards, backwards, backwards, then forwards.  See manual for more details.
    l1 = list(range(q.size))
    l2 = list(reversed(l1))
    l1 = np.array(l1 + l2 + l2 + l1)

    # Helper function which fixes all but the ith coordinate in place, and performs the objective function.
    def _fn(x, dim):
        q[dim] = x
        return func(q)

    # Performs max_its iterations of a FBBF run (see page 11)
    for _ in range(max_its):

        # A single FBBF run
        for i in l1:
            # Since _fn will automatically set values for q, no need to assign any new variables.
            optimize.brent(_fn, args=(i,), brack=I[:, i])

        # Record the output of the function after a single FBBF run, for third termination criteria.
        old_bests = np.roll(old_bests, 1)
        old_bests[0] = func(q)

        # Third termination criteria.  Exits out of the program early if proposed solutions are not
        # Being improved within a certain tolerance criteria.
        if abs((old_bests[0]-old_bests[-1])/(abs(old_bests[0]) + tol)) < tol:
            break

    # The format of g best is coordinates : f(coordinates).  Therefore, we need to append f(coordinates)
    # Onto the end before returning to keep with the format.
    return np.append(q, old_bests[0])
        

def testCCD():
    import testfuncts

    dim = 30

    ub = np.ones(dim)*100
    lb = -1*ub
    initial = np.random.rand(dim)*(ub - lb) + lb
    optimum = np.zeros(dim)

    initial = np.append(initial, 0)

    test_func = testfuncts.TestFuncts.generate_function("rosenbrock", optimum, bias=0)

    start = time.time()
    z = CCD(initial, lb, ub, alpha = 0.2, tol=0.0001, max_its = 20, third_term_its = 6, func = test_func)
    end = time.time()

    print(f"It took {end - start} seconds to run")

    print(z)

#testCCD()