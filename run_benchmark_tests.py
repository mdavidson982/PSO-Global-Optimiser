import benchmark_tests.benchmark_tests as t
import numpy as np
import testfuncts.testfuncts as tf


functions = [x for x in tf.TESTFUNCTSTRINGS if x not in t.IGNORELIST]
#functions = ["sphere", "rosenbrock"]

#t.run_benchmark_tests(replicates = 30, track_types=t.TRACK_TYPES, function_names= functions, mpso_types=t.MPSO_TYPES)
t.run_benchmark_tests(replicates = 30, track_types = t.TRACK_TYPES, function_names = functions, mpso_types = t.MPSO_TYPES)