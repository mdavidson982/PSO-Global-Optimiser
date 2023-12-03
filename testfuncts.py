import numpy as np
import parameters as p
import util as u

SPHEREID = 1
SPHERESTRING = "sphere"
ROSENBROCKID = 6
ROSENBROCKSTRING = "rosenbrock"
RASTRIGINID = 9
RASTRIGINSTRING = "rastrigin"

class TestFuncts:

    def generate_function(functionID, optimum: p.ADTYPE, bias: p.DTYPE):
        
        if type(functionID) == str:
            functionID = str.strip(functionID).lower()

        if functionID == SPHEREID or functionID == SPHERESTRING:
            func = TF._sphere_gen
        elif functionID == ROSENBROCKID or functionID == ROSENBROCKSTRING:
            func = TF._rosenbrock_gen
        elif functionID == RASTRIGINID or functionID == RASTRIGINSTRING:
            func = TF._rastrigin_gen
        else:
            raise Exception(f"functionID {functionID} does not match any available option")
        return func(optimum=optimum, bias=bias)
    
    def generate_contour(functionID, optimum: p.ADTYPE, bias: p.DTYPE, lb: p.ADTYPE, ub: p.ADTYPE) -> (p.ADTYPE, p.ADTYPE, p.ADTYPE):
        x_bound, y_bound = u.dimension_to_xy_bounds(lb, ub)
        x = np.linspace(x_bound[0], x_bound[1])
        y = np.linspace(y_bound[0], y_bound[1])
        x, y = np.meshgrid(x, y)

        if type(functionID) == str:
            functionID = str.strip(functionID).lower()

        if functionID == SPHEREID or functionID == SPHERESTRING:
            func = TF._sphere_contour
        elif functionID == ROSENBROCKID or functionID == ROSENBROCKSTRING:
            func = TF._rosenbrock_contour
        elif functionID == RASTRIGINID or functionID == RASTRIGINSTRING:
            func = TF._rastrigin_contour
        else:
            raise Exception(f"functionID {functionID} does not match any available option")
        return x, y, func(optimum=optimum, bias=bias, x=x, y=y)

    def _sphere_gen(optimum: p.ADTYPE, bias: p.DTYPE):

        #putting a test function here
        def sphere(x: p.ADTYPE) -> p.DTYPE:
            z = x - optimum
            return np.sum((z) ** 2 - bias)
        return sphere
    
    def _sphere_contour(optimum: p.ADTYPE, bias: p.DTYPE, x: p.ADTYPE, y: p.ADTYPE):
        return (x - optimum[0])**2 + y-optimum[0]**2

    #Rosenbrock
    def _rosenbrock_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rosenbrock(x) -> p.DTYPE:
            # Calculate the Rosenbrock function value for a given input x
            z = x - optimum
            return np.sum(100 * (z[:-1] - z[1:]**2)**2 + (1 - z[1:])**2)
        return rosenbrock
    
    def _rosenbrock_contour(optimum: p.ADTYPE, bias: p.DTYPE, x: p.ADTYPE, y: p.ADTYPE):
        return (x - optimum[0])* 0 + y-optimum[0]*0 #TO BE IMPLEMENTED LATER

    #Rastrigin
    def _rastrigin_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rastrigin(x) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            z = x - optimum
            return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10))
        return rastrigin
    
    def _rastrigin_contour(optimum: p.ADTYPE, bias: p.DTYPE, x: p.ADTYPE, y: p.ADTYPE):
        return (x - optimum[0])* 0 + y-optimum[0]*0 #TO BE IMPLEMENTED LATER
    
TF = TestFuncts
