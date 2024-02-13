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
ROTATEDRASTRIGINID = 10
RASTRIGINSTRING = "rotated rastrigin"

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

    #F1: Shifted Sphere Function
    def _sphere_gen(optimum: p.ADTYPE, bias: p.DTYPE):
        #putting a test function here
        def sphere(x: p.ADTYPE) -> p.DTYPE:
            #shaped_optimum = opt_reshape(x, optimum)
            #z = x - shaped_optimum

            z = x - optimum
            return np.sum((z) ** 2, axis=0) + bias
        return sphere
    
    #F2: Shifted Schwefel’s Problem 1.2
    def _shifted_schwefel_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def shifted_schwefel(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Shifted Schwefel function value for a given input x
            z = x - optimum
            return -np.sum(z * np.sin(np.sqrt(np.abs(z))))
        return shifted_schwefel
    
    #F4: F4: Shifted Schwefel’s Problem 1.2 with Noise in Fitness
    def _schwefel_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def schwefel(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Schwefel function value for a given input x
            return -np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return schwefel
    
    #F6: Shifted Rosenbrock’s Function
    def _rosenbrock_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rosenbrock(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rosenbrock function value for a given input x
            #shaped_optimum = opt_reshape(x, optimum)
            #z = x - shaped_optimum + 1

            z = x - optimum + 1
            indexes = np.arange(z.shape[0] - 1)
            return np.sum(100*(z[indexes]**2 - z[indexes+1])**2 + (z[indexes] - 1)**2, axis=0) + bias
        
        return rosenbrock

    #F7: Shifted Rotated Griewank’s Function without Bounds
    def _griewank_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def griewank(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rosenbrock function value for a given input x
            shaped_optimum = opt_reshape(x, optimum)
            z = (x - shaped_optimum)*_linearMatrix_gen(3)
            indexes = np.arange(z.shape[0])
            return np.sum(z[indexes]**2/4000) - (np.prod(np.cos(z[indexes]/np.sqrt(indexes)))) + 1 + bias
        return griewank
    
    #F8: Shifted Rotated Ackley's Function with Global Optimum on Bounds"""
    def _shifted_rotated_ackley_gen(optimum: p.ADTYPE, bias: p.DTYPE):
        def shifted_rotated_ackley_gen(x: p.ADTYPE) -> p.DTYPE:
            z = (x - optimum) * _linearMatrix_gen(100)
            d = x.shape[0]
            -20*np.exp(-0.2*np.sqrt((1/d)*(np.sum(x**2))))-np.exp((1/d)*np.sum(np.cos(2*np.pi*z)))+20+np.e + bias
        return shifted_rotated_ackley_gen
    


    #F9: Shifted Rastrigin’s Function
    def _rastrigin_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rastrigin(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            z = x - optimum
            return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10), axis=0)
        return rastrigin

    #F10: Shifted Rotated Rastrigin’s Function
    def _rotatedRastrigin_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rotatedRastrigin(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            z = (x - optimum) * _linearMatrix_gen(2)
            return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10), axis=0)
        return rotatedRastrigin
    

    
<<<<<<< HEAD
    # Shifted Schwefel's Problem F2
    def _shifted_schwefel_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def shifted_schwefel(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Shifted Schwefel function value for a given input x
            z = x - optimum
            return -np.sum(z * np.sin(np.sqrt(np.abs(z))))
        return shifted_schwefel
    
    # Schwefel's Problem 2.6 with Global Optimum on Bounds
    def _schwefel_gob_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def schwefel_gob(x: p.ADTYPE) -> p.DTYPE:
            # Calculate Schwefel's problem 2.6 function value for a given input x
            #This is found on chatgpt, is supposed to work for an n dimensional matrix
            #n should represent number of rows. Unsure if ln 122 (for i in range) line
            #is necessary but felt like it made sense 
            n = len(x)
            result = 0

            for i in range(n):
                term_i = max(abs(x[i] - 500), 0) + np.sin(np.sqrt(abs(x[i] - 500)))
                result += term_i
            return result
        return schwefel_gob
=======

    

>>>>>>> bcb4f5344efcd23a9edb9f7ae9fad29a3378047f

    
TF = TestFuncts
