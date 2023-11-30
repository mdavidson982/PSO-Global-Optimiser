import numpy as np
import parameters as p

#putting a test function here
def Sphere(array: p.ADTYPE) -> np.float64:
    return np.sum(array ** 2)

#Rosenberg
def rosenbrock(x):
    # Calculate the Rosenbrock function value for a given input x
    return np.sum(100 * (x[:-1] - x[1:]**2)**2 + (1 - x[1:])**2)

#Rastrigin
def rastrigin(x):
    # Calculate the Rastrigin function value for a given input x
    return np.sum(x**2 - 10*(np.cos(2*np.pi*x) + 10))

def 