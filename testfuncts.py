import numpy as np
import parameters as p



SPHEREID = 1
SPHERESTRING = "sphere"
ROSENBROCKID = 6
ROSENBROCKSTRING = "rosenbrock"
RASTRIGINID = 9
RASTRIGINSTRING = "rastrigin"


class TestFuncts:

    def generate_function(functionId, optimum: p.ADTYPE, bias: p.DTYPE):
        
        if type(functionId) == str:
            functionId = str.strip(functionId).lower()

        if functionId == SPHEREID or functionId == SPHERESTRING:
            return TF._sphereGenerator(optimum=optimum, bias=bias)
        if functionId == ROSENBROCKID or functionId == ROSENBROCKSTRING:
            return TF._rosenbrockGenerator(optimum=optimum, bias=bias)
        if functionId == RASTRIGINID or functionId == RASTRIGINSTRING:
            return TF._rastriginGenerator(otpimum=optimum, bias=bias)
        
        raise Exception(f"functionId {functionId} does not match any available option")

    def _sphereGenerator(optimum: p.ADTYPE, bias: p.DTYPE):
        #putting a test function here
        def sphere(x: p.ADTYPE) -> p.DTYPE:
            z = x - optimum
            return np.sum((z) ** 2 - bias)
        return sphere
    

    #Rosenberg
    def _rosenbrockGenerator(optimum:p.ADTYPE, bias: p.DTYPE):
        def rosenbrock(x) -> p.DTYPE:
            # Calculate the Rosenbrock function value for a given input x
            z = x - optimum
            return np.sum(100 * (z[:-1] - z[1:]**2)**2 + (1 - z[1:])**2)
        return rosenbrock

    #Rastrigin
    def _rastriginGenerator(optimum:p.ADTYPE, bias: p.DTYPE):
        def rastrigin(x) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            z = x - optimum
            return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10))
        return rastrigin
    
TF = TestFuncts
