import numpy as np
from scipy import optimize
import parameters as p

def _testFunct(arr: p.ADTYPE):
    return sum(arr**2)

def CCD(initial: p.DTYPE, lb: p.ADTYPE, ub: p.ADTYPE, 
        alpha: p.DTYPE, tol: p.DTYPE, max_its: int, third_term_its: int, func):
    omega = alpha*(ub-lb) # Set restricted domain, as per step 2 on page 11 of manual

    old_bests = np.finfo(p.DTYPE).max*np.ones(third_term_its, dtype=p.DTYPE)

    # Set I low and I high, as per step 3 on page 11 of manual
    I_low: np.ndarray = np.max(np.vstack((lb[np.newaxis], initial[np.newaxis] - omega)), axis=0)
    I_high: np.ndarray = np.min(np.vstack((ub[np.newaxis], initial[np.newaxis] + omega)), axis=0)
    I = np.vstack((I_low[np.newaxis], I_high[np.newaxis]))
    
    q = initial.copy() #Copy so that initial is not overwritten

    # l1 indicates the sequence of coordinates to perform brent on.  As per Step F, 
    # CCD should be performed forwards, backwards, backwards, then forwards.  See manual for more details.
    l1 = list(range(q.size))
    l2 = list(reversed(l1))
    l1 = l1 + l2 + l2 + l1

    # Performs max_its iterations of a FBBF run (see page 11)
    for _ in range(max_its):
        for i in l1:
            def fn(x):
                q[i] = x
                return func(q)
            
            optimize.brent(fn, brack=I[:, i])
        old_bests = np.roll(old_bests, 1)
        old_bests[0] = func(q)

        if abs(old_bests[0]-old_bests[-1])/(abs(old_bests[0]) + tol) < tol:
            break
    return q
        

def testCCD():
    CCD(np.array((1, 2, 3)), np.array((-100, -100, -100)), np.array((100, 100, 100)), 
        alpha = 0.2, tol=0.0001, max_its = 20, third_term_its = 6, func = _testFunct)

testCCD()