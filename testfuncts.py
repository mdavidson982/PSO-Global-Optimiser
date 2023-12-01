import numpy as np
import parameters as p


def sphereGenerator(optimum: p.ADTYPE):
    #putting a test function here
    def Sphere(x: p.ADTYPE) -> np.float64:
        z = x - optimum
        return np.sum((z) ** 2)
    return Sphere

#Rosenberg
def rosenbrockGenerator(optimum:p.ADTYPE):
    def rosenbrock(x):
        # Calculate the Rosenbrock function value for a given input x
        z = x - optimum
        return np.sum(100 * (z[:-1] - z[1:]**2)**2 + (1 - z[1:])**2)
    return rosenbrock

#Rastrigin
def rastriginGenerator(optimum:p.ADTYPE):
    def rastrigin(x):
        # Calculate the Rastrigin function value for a given input x
        z = x - optimum
        return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10))
    return rastrigin