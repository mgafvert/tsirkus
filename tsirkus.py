import numpy as np
from numpy.linalg import matrix_power
import pdb



class Tsirkus:
    """
    A class to represent a Tsirkus board game configuration. Tsirkus is a 
    classic Estonian version of the worlwide classic "Snakes and Ladders" 
    board game originating from ancient India (https://en.wikipedia.org/wiki/Snakes_and_ladders).
    """
    def __init__(self, 
                 jumps = [(3,19), (9,11), (14,30), (15,7), (17,36), (20,42), (26,35), (34,97), (38,57), (50,32), (59,79), (64,78), (68,48), (70,72), (77,98), (92,74), (100,82), (103,96), (107,23), (108,114), (112,91), (116,105), (119,101)], 
                 shape = (12,10)):
        self.shape = shape
        self.N = shape[0]*shape[1]
        # setup game path
        self.jumps = np.array(list(zip(*jumps))) - 1
        self.path = np.arange(self.N)
        # check inputs
        if self.jumps.size > 0:
            if not (np.amax(self.jumps) < self.N): 
                raise ValueError(f"Jumps must be from/to positions less than {self.N}!")
            if not (np.amin(self.jumps) >= 0):
                raise ValueError(f"Jumps must be from/to positions greater than 0!")
            if  not (len(np.unique(self.jumps[0])) == len(self.jumps[0])):
                raise ValueError(f"Jumps from position must be unique!")
            self.path[self.jumps[0]] = self.jumps[1]
        # setup game matrix
        self.dice = 6 # one 6-sided dice (consider generalizing to m n-sided dice)
        self.dice_probs = np.ones(self.dice)/self.dice
        self.P = np.zeros((self.N, self.N))
        for i in range(self.N-1): 
            if self.path[i] != i: # skip jump positions (zero prob to stay there)
                continue
            for j in range(self.dice):
                if i+j+1 < self.N:
                    self.P[i, self.path[i+j+1]] += self.dice_probs[j]
                else:
                    self.P[i, self.path[self.N-1 - (i+j+1 - (self.N-1))]] += self.dice_probs[j]
        self.P[-1,-1] = 1 # stick at last position (goal)


    def __repr__(self):
        return f"Tsirkus board game configuration with {len(self.path)} steps, {len(self.jumps)} jumps and shape {self.shape}"
        
t = Tsirkus()
#pdb.set_trace()
        




