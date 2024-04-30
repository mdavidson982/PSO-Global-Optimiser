import numpy as np
import utils.parameters as p
import utils.util as u
import utils.consts as c
from pso.psodataclass import FunctionData

SPHEREID = 1
SPHERESTRING = "sphere"

SHIFTEDSCHWEFELID = 2
SHIFTEDSCHWEFELSTRING = "shiftedschwefel"

SHIFTEDELLIPTICID = 3
SHIFTEDELLIPTICSTRING = "shiftedelliptic"

SCHWEFELID = 4
SCHWEFELSTRING = "schwefel"

SCHWEFELGOBID = 5
SCHWEFELGOBSTRING = "schwefelgob"

ROSENBROCKID = 6
ROSENBROCKSTRING = "rosenbrock"

GRIEWANKID = 7
GRIEWANKSTRING = "griewank"

SHIFTEDROTATEDACKLEYID = 8
SHIFTEDROTATEDACKLEYSTRING = "shiftedrotatedackley"

RASTRIGINID = 9
RASTRIGINSTRING = "rastrigin"

ROTATEDRASTRIGINID = 10
ROTATEDRASTRIGINSTRING = "rotatedrastrigin"

TESTFUNCTIDS = [SPHEREID, SHIFTEDSCHWEFELID, SHIFTEDELLIPTICID, SCHWEFELID,
                    SCHWEFELGOBID, ROSENBROCKID, GRIEWANKID, SHIFTEDROTATEDACKLEYID,
                    RASTRIGINID, ROTATEDRASTRIGINID]

TESTFUNCTSTRINGS = [SPHERESTRING, SHIFTEDSCHWEFELSTRING, SHIFTEDELLIPTICSTRING, SCHWEFELSTRING,
                    SCHWEFELGOBSTRING, ROSENBROCKSTRING, GRIEWANKSTRING, SHIFTEDROTATEDACKLEYSTRING,
                    RASTRIGINSTRING, ROTATEDRASTRIGINSTRING]


def opt_reshape(x: p.ADTYPE, optimum: p.ADTYPE):
    n1 = len(x.shape)
    n2 = len(optimum.shape)
    new_shape = optimum.shape + ((1,) * (n1-n2))
    return optimum.reshape(new_shape)

#Function used in Griewank test, placeholder values, code was created with ChatGPT
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

def gram_schmidt(A):
    """
    Perform the Gram-Schmidt process on a matrix A.
    Returns an orthonormal basis for the column space of A.
    """
    Q, R = np.linalg.qr(A)
    return Q

def create_orthogonal_matrix(n):
    """
    Create an n by n orthogonal matrix.
    """
    A = np.random.rand(n, n)  # Generating a random matrix
    orthogonal_matrix = gram_schmidt(A)
    return orthogonal_matrix

class TestFuncts:

    def generate_function(
            functionID: str | int = None, 
            optimum: p.ADTYPE = None, 
            bias: p.DTYPE = None,
            domaindata: FunctionData = None,
        ):
        if domaindata is not None:
            return TestFuncts._generate_function_helper(
                functionID = functionID,
                optimum = domaindata.optimum,
                bias = domaindata.bias
            )
        else:
            return TestFuncts._generate_function_helper(
                functionID = functionID,
                optimum = optimum,
                bias = bias
            )

    def _generate_function_helper(functionID, optimum: p.ADTYPE, bias: p.DTYPE):
        if type(functionID) == str:
            functionID = str.strip(functionID).lower()

        if functionID == SPHEREID or functionID == SPHERESTRING:
            func = TF._sphere_gen
        elif functionID == SHIFTEDSCHWEFELID or functionID == SHIFTEDSCHWEFELSTRING:
            func = TF._schwefel_gen
        elif functionID == SHIFTEDELLIPTICID or functionID == SHIFTEDELLIPTICSTRING:
            func = TF._shifted_elliptic_gen
        elif functionID == SCHWEFELID or functionID == SCHWEFELSTRING:
            func = TF._schwefel_gen
        elif functionID == SCHWEFELGOBID or functionID == SCHWEFELGOBSTRING:
            func = TF._schwefel_gob_gen
        elif functionID == ROSENBROCKID or functionID == ROSENBROCKSTRING:
            func = TF._rosenbrock_gen
        elif functionID == GRIEWANKID or functionID == GRIEWANKSTRING:
            func = TF._griewank_gen
        elif functionID == SHIFTEDROTATEDACKLEYID or functionID == SHIFTEDROTATEDACKLEYSTRING:
            func = TF._shifted_rotated_ackley_gen
        elif functionID == RASTRIGINID or functionID == RASTRIGINSTRING:
            func = TF._rastrigin_gen
        elif functionID == ROTATEDRASTRIGINID or functionID == ROTATEDRASTRIGINSTRING:
            func = TF._rotatedRastrigin_gen
        else:
            raise Exception(f"functionID {functionID} does not match any available option")
        return func(optimum=optimum, bias=bias)
    
    def generate_contour(function, lb: p.ADTYPE, ub: p.ADTYPE) -> (p.ADTYPE, p.ADTYPE, p.ADTYPE):
        x = np.linspace(lb[0], ub[1], 500)
        y = np.linspace(lb[0], ub[1], 500)
        X, Y = np.meshgrid(x, y)
        z = np.array((X, Y))
        Z = np.apply_along_axis(function, axis = 0, arr = z)
        return X, Y, Z

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

            # Tile the array by len(array) times.  Then, take the lower triangluar matrix.
            # This is done because schwefel continually adds sums up to the ith index,
            # So lower triangular matrices will automatically zero out any other values.
            z = x - optimum
            length = len(z)
            new = np.tile(z, length).reshape((length, length))
            lower = np.tril(new)
            return np.sum(np.sum(lower, axis=1)**2) + bias
        return shifted_schwefel
    
    #F3: Shifted Rotated High Condition Elliptic
    def _shifted_elliptic_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def shifted_elliptic(x: p.ADTYPE) -> p.DTYPE:
            orth_matrix = create_orthogonal_matrix(len(10)) #Just 10 by 10 for now. We don't have examples of how the matrix should be look.
            shaped_optimum = opt_reshape(x, optimum)
            z = (x - shaped_optimum)
            indexes = np.arange(z.shape[0] - 1)
            return np.sum(((10**6)**(indexes/(len(z)-1)))*(z[indexes])**2) + bias
        return shifted_elliptic
            
    #F4: F4: Shifted Schwefel’s Problem 1.2 with Noise in Fitness
    def _schwefel_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def schwefel(x: p.ADTYPE) -> p.DTYPE:
            z = x - optimum
            length = len(z)
            new = np.tile(z, length).reshape((length, length))
            lower = np.tril(new)
            prod_1 = np.sum(np.sum(lower, axis=1)**2)
            prod_2 = 1 + 0.4*np.abs(np.random.standard_normal(1))
            return prod_1*prod_2

        return schwefel
    
    #F5 Schwefel's Problem 2.6 with Global Optimum on Bounds
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
            z = (x - shaped_optimum)*_linearMatrix_gen(3)
            indexes = np.arange(z.shape[0])
            return np.sum(z[indexes]**2/4000) - (np.prod(np.cos(z[indexes]/np.sqrt(indexes)))) + 1 + bias
        return griewank
    
    #F8: Shifted Rotated Ackley's Function with Global Optimum on Bounds"""
    def _shifted_rotated_ackley_gen(optimum: p.ADTYPE, bias: p.DTYPE):
        def shifted_rotated_ackley_gen(x: p.ADTYPE) -> p.DTYPE:
            z = (x - optimum) * _linearMatrix_gen(100)
            d = x.shape[0]
            return -20*np.exp(-0.2*np.sqrt((1/d)*(np.sum(x**2))))-np.exp((1/d)*np.sum(np.cos(2*np.pi*z)))+20+np.e + bias
        return shifted_rotated_ackley_gen
    
    
    #F9: Shifted Rastrigin’s Function
    def _rastrigin_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rastrigin(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            z = x + np.expand_dims(x, list(range(1, len(optimum.shape)+1)))
            z = x - optimum
            return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10), axis=0) + bias
        return rastrigin

    #F10: Shifted Rotated Rastrigin’s Function
    def _rotatedRastrigin_gen(optimum:p.ADTYPE, bias: p.DTYPE):
        def rotatedRastrigin(x: p.ADTYPE) -> p.DTYPE:
            # Calculate the Rastrigin function value for a given input x
            z = (x - optimum) * _linearMatrix_gen(2)
            return np.sum(z**2 - 10*(np.cos(2*np.pi*z) + 10), axis=0)
        return rotatedRastrigin
    

    
TF = TestFuncts
