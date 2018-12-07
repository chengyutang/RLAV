import numpy as np

class State(object):
    
    def __init__(self, crd, v, dists, destPos):
        self.crd = crd #np.array([x, y])
        self.v = v
        self.dists = dists #np.array([front, right, left]), front visibility:2, left&right visibility:1
        self.destPos = destPos # relative position of destination

    # make objects comparable by speed and perception of its surroundings
    # the agent doesn't know its coordinates
    def __eq__(self, other):
        # return np.all(self.dists == other.dists) and self.v == other.v and np.all(self.destPos == other.destPos)
        if isinstance(other, self.__class__):
            return np.all(self.dists == other.dists) #and np.all(self.destPos == other.destPos)
        return False

    def __hash__(self):
        # return hash(tuple(self.dists) + (self.v, ) + tuple(self.destPos))
        # return hash(tuple(self.dists) + tuple(self.destPos))
        return hash(tuple(self.dists))