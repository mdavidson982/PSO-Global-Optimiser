import numpy as np
import parameters as p
import util as u
import consts as c

SPHEREID = 1
SPHERESTRING = "sphere"
ROSENBROCKID = 6
ROSENBROCKSTRING = "rosenbrock"
GRIEWANKSTRING = "griewank"
GRIEWANKID = 7
RASTRIGINID = 9
RASTRIGINSTRING = "rastrigin"

def opt_reshape(x: p.ADTYPE, optimum: p.ADTYPE):
    n1 = len(x.shape)
    n2 = len(optimum.shape)
    new_shape = optimum.shape + ((1,) * (n1-n2))
    return optimum.reshape(new_shape)

#Function used in Griewank test, placeholder values as Danh is big bozo
def _linearMatrix_gen(conditionNum):
    # Generate a diagonal scaling matrix Bruh1
    singular_values = np.linspace(1, conditionNum, p.NUM_DIM)
    Bruh1 = np.diag(singular_values)
    # Generate a random orthogonal matrix Bruh3 using QR decomposition
    Bruh2 = np.random.randn(p.NUM_DIM, p.NUM_DIM)
    Bruh3 = np.linalg.qr(Bruh2)
    # Construct the rotation matrix Bruh4 = Bruh3^T * Bruh1 * Bruh2
    Bruh4 = np.dot(np.dot(Bruh3.T, Bruh1), Bruh3)
    return Bruh4

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
    
    def generate_contour(function, lb: p.ADTYPE, ub: p.ADTYPE) -> (p.ADTYPE, p.ADTYPE, p.ADTYPE):
        x_bound, y_bound = u.dimension_to_xy_bounds(lb, ub)
        x = np.linspace(x_bound[0], x_bound[1], 500)
        y = np.linspace(y_bound[0], y_bound[1], 500)
        X, Y = np.meshgrid(x, y)
        input = np.array((X, Y))
        return X, Y, function(input)

    def _sphere_gen(optimum: p.ADTYPE, bias: p.DTYPE):
        #putting a test function here
        def sphere(x: p.ADTYPE) -> p.DTYPE:
            #shaped_optimum = opt_reshape(x, optimum)
            #z = x - shaped_optimum

            z = x - optimum
            return np.sum((z) ** 2, axis=0) + bias
        return sphere

    #Griewank lol, NEEDS REVIEW 
    def _griewank_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def griewank(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rosenbrock function value for a given input x
            shaped_optimum = opt_reshape(x, optimum)
            z = (x - shaped_optimum)*_linearMatrix_gen(3)
            indexes = np.arange(z.shape[0])
            return np.sum(z[indexes]**2/4000) - (np.prod(np.cos(z[indexes]/np.sqrt(indexes)))) + 1 + bias
        return griewank
    
    #Rosenbrock
    def _rosenbrock_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rosenbrock(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rosenbrock function value for a given input x
            #shaped_optimum = opt_reshape(x, optimum)
            #z = x - shaped_optimum + 1

            z = x - optimum + 1
            indexes = np.arange(z.shape[0] - 1)
            return np.sum(100*(z[indexes]**2 - z[indexes+1])**2 + (z[indexes] - 1)**2, axis=0) + bias
        
        return rosenbrock

    #Rastrigin
    def _rastrigin_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rastrigin(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            z = x - optimum
            return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10), axis=0)
        return rastrigin
    
    def _schwefel_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def schwefel(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            shaped_optimum = opt_reshape(x, optimum) 
            return -np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return schwefel
    
    """F8: Shifted Rotated Ackley's Function with Global Optimum on Bounds"""
    def _shifted_rotated_ackley_gen(optimum: p.ADTYPE, bias: p.DTYPE):
        def shifted_rotated_ackley_gen(x: p.ADTYPE) -> p.DTYPE:
            z = (x - optimum) * _linearMatrix_gen(100)
            d = x.shape[0]
            -20*np.exp(-0.2*np.sqrt((1/d)*(np.sum(x**2))))-np.exp((1/d)*np.sum(np.cos(2*np.pi*z)))+20+np.e + bias

    
TF = TestFuncts
