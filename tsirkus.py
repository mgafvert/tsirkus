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
        """
        jump: list of tuples of jump positions (from, to) where from and to are board positions 1..N
        shape: tuple of board shape (rows, columns) where game path length N = rows*columns
        """
        self.shape = shape
        self.N = shape[0]*shape[1]
        # setup game path
        self.jumps = np.array(list(zip(*jumps))) - 1 # -1 to translate from board positions 1..N to array indices 0..N-1 
        self.path = np.arange(self.N) # enumerate board positions
        # check inputs
        if self.jumps.size > 0: # if any jumps 
            if not (np.amax(self.jumps) < self.N - 1): 
                raise ValueError(f"Jumps must be from/to positions less than {self.N}!")
            if not (np.amin(self.jumps) >= 0):
                raise ValueError(f"Jumps must be from/to positions greater than 1!")
            if  not (len(np.unique(self.jumps[0])) == len(self.jumps[0])):
                raise ValueError(f"Jumps from position must be unique!")
            if set(self.jumps[0]).intersection(set(self.jumps[1])):
                raise ValueError(f"Jumps from and to positions must be distinct!")
            self.path[self.jumps[0]] = self.jumps[1] # jumps from i to j encoded as path[i] = j
        # setup game matrix
        self.dice = 6 # one 6-sided dice (consider generalizing to m n-sided dice)
        self.dice_probs = np.ones(self.dice)/self.dice
        self.P = np.zeros((self.N, self.N))
        for i in range(self.N-1): # skip end position since not moving from there
            if self.path[i] != i: # skip jump positions (zero prob to stay there)
                continue
            for j in range(self.dice):
                if i+j+1 < self.N:
                    self.P[i, self.path[i+j+1]] += self.dice_probs[j]
                else:
                    self.P[i, self.path[self.N-1 - (i+j+1 - (self.N-1))]] += self.dice_probs[j]
        self.P[-1,-1] = 1 # stick at end position

    def evolve_final(self, p0 = None, p_final = 1.):
            """
            Generator to evolve game from initial state p0 until p[-1] >= p_final (probability at game path end position)
            yields position probability vector p at each step 
            p0 is initial state (defaults to 1.0 at start position)
            p_final is end position probability condition (defaults to 1.0 and infinite game if back jumps are present)
            """
            if p0 is None:
                p0 = np.eye(self.N,1)
            p = p0
            yield p
            while p[-1] < p_final:
                p = self.P.T@p
                yield p
            
    def evolve_turns(self, p0 = None, N=1):
            if p0 is None:
                p0 = np.eye(self.N,1)
            return matrix_power(self.P.T,N)@p0

    def __repr__(self):
        return f"Tsirkus board game configuration with {len(self.path)} steps, {len(self.jumps)} jumps and shape {self.shape}"
        
t = Tsirkus()
#pdb.set_trace()
        


if __name__ == '__main__':
    t = Tsirkus()
    p0 = np.eye(t.N,1)
    p_final = 0.5
    for i, p in enumerate(t.evolve_final(p0, p_final)):
        pass
    print(i,p[-1])
 
    P = np.squeeze(np.array([p for p in t.evolve_final(p0, p_final)]))
    p_end = P[:,-1]

