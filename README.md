# PSO-GLOBAL OPTIMIZER
The package implements a novel optimization method, called MPSO-CCD (Multiple Particle Swarm Optimizer - Cyclic Coordinte Descent).

## Installation
run pip install requirements.txt to install all necessary packages.  It is generally good practice to use a specific virtual environment or conda environment when installing python packages.

## Use
The library has multiple different components, each with distinct uses.  In order of importance, their use is listed below.

### mpso_ccd
The module contains classes and methods pertaining to the mpso-ccd optimizer, and is the basis of the package.  

The submodule psofuncts contains the methods responsible for the operation of pso and ccd.  These functions are used by the MPSO and PSO classes to control the execution of PSO.  They can be used independently of the PSO and MPSO objects, but this is not recommended.

psodataclass.py:  File which contains dataclasses that define the operation of MPSO and PSO.  For instance, the class PSOHyperparameters is a dataclass which defines the hyperparameters upon which the PSO operation runs.  These classes are all json-serializable, and can be read in from files or written to files using the codec.  These dataclasses will become the basis upon which the PSO and MSPO objects are built.

codec.py:  File which allows for json encoding and decoding of classes in psodataclass.  dataclass_to_json_file() writes any dataclass to a json file.  json_file_to_dataclass() reads in a dataclass from a json file and handles types.

pso.py:  Implements two classes, PSO and PSOLogger.  PSOLogger is a thin wrapper around PSO, which is only responsible for collecting intermittent values that PSO produces.  PSO is a class which has several methods that allow for running PSO, mainly run_pso().  PSO needs to be supplied with several dataclasses (see above) in order to properly function.  These dataclasses can either be hard-coded in, or read via the codec.  Many have default values, though the best values will be problem-dependent.  For reference on how to create a pso object, see mpso_ccd/test/test_pso.py.

mpso.py:  Implements two classes, MPSO and MPSOLogger.  MPSOLogger is a thin wrapper around MPSO, which is only responsible for collecting intermittent values that MPSO produces.  It works very similarly to PSOLogger. Much like PSO, MPSO needs to be supplied with a PSO object and several other optional dataclasses which define the operation of MPSO.  The best values will be problem dependent, so tuning may be required.  For reference on how to create an mpso object, see mpso_ccd/test/test_mpso.py.

### testfuncts

Several test functions have been created for use in benchmark testing.  However, if more functions are required, a string and integer id must be created.  For reference, look at how sphere is created.  

### benchmark_tests

Benchmark tests is responsible for capturing the performance of MPSO-CCD in a data-driven fashion.  For any number of functions in testfuncts, it can run a test with MPSO or a test with MPSO that utilizes the local optimizer CCD.  It can be configured to track time and/or quality.  Each individual configuration can be replicated many times, which helps track statistically how well the configuration is performing.  To run a benchmark test, run the file run_benchmark_tests.py.  Edit the parameters as needed, and find the results in benchmark_tests/benchmarkruns.  Each individual function has specific values which control their behavior in the jso files of benchmark_tests/configs.  These may be manually changed or changed via the helper functions in edit_benchmark_tests.py (see below)

If visualizations of the output are required, the file make_visualizations will be useful.  Line graphs of the quality or time over iterations for both PSO and MPSO may be created, as well as boxplots and histograms of the replicates.  First, Run a benchmark test and provide the necessary path in the path variable, and function if required.  Then, navigate to the folder that you have supplied and copy your graphs.

If the parameters for any of the benchmark test functions need to be changed, look at edit_benchmark_tests.py.  To edit one function, first determine the dataclass's parameters that you would like to be changed.  In the new dict, edit as needed.  Then, run run_edit_specific_function().  If you need to edit multiple functions at once, 

### visualization

This module 


Structure of various variables:

upper_bound:
[num_dim x 1] matrix.  Stores the values that maintain the upper bound of the domain

lower_bound:
[num_dim x 1] matrix.  Stores the values htat maintain the lower bound of the domain

pos_matrix:
[num_dim X num_part] matrix.  Every column is a particle, and every row is a dimension.  So, every column has the x, y, z etc. coordinates of the particle.


vel_matrix:
[num_dim X num_part] matrix.  

p_best:
[num_dim + 1 X num_part] matrix.  Stores the coordinates of every particle's personal best, along with the evaluated result on the bottom. So,

    x1: | | | |   <- this upper part has the same dimensions as the pos and vel matrices
    x2: | | | |
    ... 
    xn: | | | |
        _______
result: | | | |

g_best:
[num_dim X 1] matrix.  Stores the global best

v_max:
[num_dim X 1] matrix.  Maximum velocity of any given particle, given by alpha(U - L).