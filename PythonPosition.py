#Outdated File. Not Used. Pretty Cringe.

#Comsc490 Project PSO Position
import random
import numpy as np
g_best = "bruh"; #matrix, vector (global best)
p_best = "bruh"; #matrix, m*n (personal best)
num_part = ["bruh","bruh"]; #int (range)
num_dim = "bruh"; #int
v_part = "bruh"; #matrix m*n (velocity)
x_pos = "bruh"; #matrix m*n
LowerBound = 0; #float
UpperBound = 100; #float
LowerPartBound = 300; #int ~300
UpperPartBound = 600; #int ~600



class Particle:
    def __init__(self):
        self.generateLength()
        self.position = self.generate_array()
        self.velocity = self.generate_array()
        
        
    def generateLength(self):
        self.length = random.randint(LowerPartBound, UpperPartBound) #generate a random number between 300 and 600
        
    
    def generate_array(self):
        
        return np.random.uniform(LowerBound, UpperBound, size = self.length) 
      
    def display(self):
        print("Position: " + str(self.position)+"\n")
        print("Velocity: " + str(self.velocity)+"\n")


particle = Particle()


#Generate random amount of particles

list = []

for x in range(6):
    d = {}
    d["particle{0}".format(x)]= Particle();
    list.append(d["particle"+str(x)])
