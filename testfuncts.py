import numpy as np
import parameters as p
import util as u
import consts as c

SPHEREID = 1
SPHERESTRING = "sphere"
ROSENBROCKID = 6
ROSENBROCKSTRING = "rosenbrock"
RASTRIGINID = 9
RASTRIGINSTRING = "rastrigin"

def _optimumShift(optimum: p.ADTYPE, x: p.ADTYPE, y: p.ADTYPE) -> (p.ADTYPE, p.ADTYPE):
    "Utililty function for contour plots.  Shifts x & y by the optimum."
    return x - optimum[c.XDIM], y - optimum[c.YDIM]

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
        x = np.linspace(x_bound[0], x_bound[1], 500)
        y = np.linspace(y_bound[0], y_bound[1], 500)
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
            return np.sum((z) ** 2) + bias
        return sphere
    
    def _sphere_contour(optimum: p.ADTYPE, bias: p.DTYPE, x: p.ADTYPE, y: p.ADTYPE):
        x, y = _optimumShift(optimum, x, y)
        return (x)**2 + (y)**2 + bias

    #Rosenbrock
    def _rosenbrock_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rosenbrock(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rosenbrock function value for a given input x
            z = x - optimum + 1
            indexes = np.arange(z.shape[0] - 1)
            return np.sum(100*(z[indexes]**2 - z[indexes+1])**2 + (z[indexes] - 1)**2) + bias
        
        return rosenbrock
    
    def _rosenbrock_contour(optimum: p.ADTYPE, bias: p.DTYPE, x: p.ADTYPE, y: p.ADTYPE):
        x, y = _optimumShift(optimum, x, y)
        x = x + 1 # Part of the rosenbrock shift
        y = y + 1 # Part of the rosenbrock shift
        return 100*(x**2 - y)**2 + (x - 1)**2 + bias

    #Rastrigin
    def _rastrigin_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rastrigin(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            z = x - optimum
            return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10))
        return rastrigin
    
    def _rastrigin_contour(optimum: p.ADTYPE, bias: p.DTYPE, x: p.ADTYPE, y: p.ADTYPE):
        return (x - optimum[0])* 0 + y-optimum[0]*0 #TO BE IMPLEMENTED LATER
    
TF = TestFuncts
