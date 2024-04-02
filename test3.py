import os
import shutil
import testfuncts.testfuncts as tf
import json
import numpy as np
import pso.codec as codec

mypath = os.path.join("benchmark_tests", "configs")
files = os.listdir(mypath)
for funct in tf.TESTFUNCTSTRINGS:
    print(funct)
    functpath = os.path.join(mypath, funct)
    print(os.path.exists(functpath))
    print(os.path.exists(os.path.join(functpath, "domain_data.json")))
    
    with open(os.path.join(functpath,"domain_data.json")) as file:
        z = json.load(file)
    z["optimum"] = [0]*30
    z["bias"] = 0
    with open(os.path.join(functpath, "domain_data.json"), "w+") as file:
        json.dump(z, file)